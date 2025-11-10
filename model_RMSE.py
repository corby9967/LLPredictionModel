#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model.py — ΔSL(5) 예측 + ΔLL(=ΣΔSL) 합-방식, DLL 보조헤드, '검증 Loss' 기준 체크포인트
- 입력: (5레벨×12) + 글로벌4 = 64차원 [기본]. --replicate_globals 로 80차원 전환 가능
- 손실:
    L_SL  = Huber(beta_sl) on SL_post  (레벨별 λ 가중 적용; 합=1일 필요 없음, 각 λ∈[0,1])
    L_LL  = log-cosh on LL_post (합-방식 dll_sum 사용)
    L_cons= MSE(dll_aux, dll_sum.detach())  # 합-보조 일관성 soft 제약
    L_tv  = 인접 레벨 ΔSL 스무딩(옵션; tv_w가 0이면 꺼짐)
    total = wSL*L_SL + wLL*L_LL + wC*L_cons + tv_w*L_tv
- 체크포인트/조기종료: **검증 total loss 기준**
- CSV: rmse_dll, rmse_dsl_all, per-level rmse, best_epoch, lambdas 기록
"""

import json, argparse, random, csv, time, os
from pathlib import Path
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

LEVELS = 5

# -------------------
# 유틸
# -------------------
def set_seed(seed: int = 42):
    """랜덤 시드 고정 (완전한 재현성)"""
    if seed is None: return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # CUDA 완전 재현성 보장
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # PyTorch의 완전 결정론적 알고리즘 사용(옵션).
    # 이 기능을 활성화하려면 프로세스 시작 전에 환경변수
    # CUBLAS_WORKSPACE_CONFIG=':4096:8' 를 설정해야 합니다.
    # (설정하지 않으면 torch.use_deterministic_algorithms(True)에서
    #  RuntimeError 발생할 수 있음)
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG'):
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:
            # 오류가 나더라도 진행(환경/버전 문제일 수 있음)
            print(f"Warning: unable to enable torch.use_deterministic_algorithms: {e}")
    else:
        # 사용자에게 안내(환경변수를 설정하면 CUDA 레이어까지 결정론적)
        pass

def split_indices(n, seed=None, ratios=(0.7, 0.15, 0.15)):
    ids = list(range(n))
    rnd = random.Random(seed); rnd.shuffle(ids)
    n_tr = int(n * ratios[0]); n_va = int(n * ratios[1])
    tr = ids[:n_tr]; va = ids[n_tr:n_tr+n_va]; te = ids[n_tr+n_va:]
    return tr, va, te

def rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((a - b) ** 2)).item())

# -------------------
# 데이터셋 (64/80 차원)
# -------------------
class SpineDataset(Dataset):
    def __init__(self, raw_path: str, gt_path: str, replicate_globals: bool = False, zscore: bool = True):
        with open(raw_path, 'r', encoding='utf-8') as f:
            self.raw: Dict[str, Dict] = json.load(f)
        with open(gt_path, 'r', encoding='utf-8') as f:
            self.gt: Dict[str, Dict] = json.load(f)

        raw_pids = set(self.raw.keys())
        gt_pids = {k.split('_')[0] for k in self.gt.keys()
                   if k.endswith('_10000') and f"{k.split('_')[0]}_20000" in self.gt}
        self.pids = sorted(list(raw_pids & gt_pids), key=lambda x: int(x) if str(x).isdigit() else x)

        self.replicate_globals = replicate_globals
        self.zscore = zscore

        self.X, self.y_sl_pre, self.y_sl_post, self.y_ll_pre, self.y_ll_post = self._build_all()
        self.mu, self.sd = None, None
        if self.zscore and len(self.X) > 0:
            self.mu = self.X.mean(axis=0)
            self.sd = self.X.std(axis=0); self.sd[self.sd == 0] = 1.0
            self.X = (self.X - self.mu) / self.sd

    def _vec5(self, d: Dict, key: str, default=0.0) -> np.ndarray:
        v = d.get(key, [default] * LEVELS)
        if not isinstance(v, list):
            return np.array([default]*LEVELS, dtype=np.float32)
        v = [(default if (x is None) else x) for x in v]
        return np.array((v + [default]*LEVELS)[:LEVELS], dtype=np.float32)

    def _one(self, pid: str):
        r = self.raw.get(pid, {})
        pre = self.gt.get(f"{pid}_10000", {})
        post = self.gt.get(f"{pid}_20000", {})
        if not r or not pre or not post: return None

        sl_pre  = self._vec5(r, 'sl_pre')
        op_tlif = self._vec5(r, 'op_tlif')
        op_plif = self._vec5(r, 'op_plif')
        op_olif = self._vec5(r, 'op_olif')
        op_alif = self._vec5(r, 'op_alif')
        cage_w  = self._vec5(r, 'cage_w')
        cage_l  = self._vec5(r, 'cage_l')
        cage_h  = self._vec5(r, 'cage_h')
        cage_d  = self._vec5(r, 'cage_d')
        pos_a   = self._vec5(r, 'cagepos_anterior')
        pos_c   = self._vec5(r, 'cagepos_center')
        pos_p   = self._vec5(r, 'cagepos_posterior')

        X_levels = np.stack([
            sl_pre,
            op_tlif, op_plif, op_olif, op_alif,
            cage_w, cage_l, cage_h, cage_d,
            pos_a, pos_c, pos_p
        ], axis=1).astype(np.float32)  # (5,12)

        g = np.array([
            float(r.get('age', 0.0) or 0.0),
            float(r.get('bmi', 0.0) or 0.0),
            float(r.get('sex', 0) or 0),
            float(r.get('ss_pre', 0.0) or 0.0),
        ], dtype=np.float32)

        if self.replicate_globals:
            G = np.repeat(g[None, :], repeats=LEVELS, axis=0)  # (5,4)
            X = np.concatenate([X_levels, G], axis=1).reshape(-1)  # (80,)
        else:
            X = np.concatenate([X_levels.reshape(-1), g], axis=0)  # (64,)

        y_sl_pre  = np.array([float(pre.get(f"SL{i+1}", 0.0) or 0.0) for i in range(LEVELS)], dtype=np.float32)
        y_sl_post = np.array([float(post.get(f"SL{i+1}", 0.0) or 0.0) for i in range(LEVELS)], dtype=np.float32)
        y_ll_pre  = float(pre.get("LL", 0.0) or 0.0)
        y_ll_post = float(post.get("LL", 0.0) or 0.0)
        return X.astype(np.float32), y_sl_pre, y_sl_post, y_ll_pre, y_ll_post

    def _build_all(self):
        Xs, slp, slq, llp, llq = [], [], [], [], []
        for pid in self.pids:
            one = self._one(pid)
            if one is None: continue
            X, a, b, c, d = one
            Xs.append(X); slp.append(a); slq.append(b); llp.append(c); llq.append(d)
        X = np.vstack(Xs).astype(np.float32) if Xs else np.zeros((0, 64), dtype=np.float32)
        y_sl_pre  = np.vstack(slp).astype(np.float32) if slp else np.zeros((0, LEVELS), dtype=np.float32)
        y_sl_post = np.vstack(slq).astype(np.float32) if slq else np.zeros((0, LEVELS), dtype=np.float32)
        y_ll_pre  = np.array(llp, dtype=np.float32) if llp else np.zeros((0,), dtype=np.float32)
        y_ll_post = np.array(llq, dtype=np.float32) if llq else np.zeros((0,), dtype=np.float32)
        return X, y_sl_pre, y_sl_post, y_ll_pre, y_ll_post

    def __len__(self): return self.X.shape[0]

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.X[idx]),
            torch.from_numpy(self.y_sl_pre[idx]),
            torch.from_numpy(self.y_sl_post[idx]),
            torch.tensor(self.y_ll_pre[idx], dtype=torch.float32),
            torch.tensor(self.y_ll_post[idx], dtype=torch.float32),
        )

    @property
    def in_dim(self) -> int: return self.X.shape[1]

# -------------------
# 모델: 백본 + (DSL 5) + (DLL 보조 1)
# -------------------
class SharedMLP(nn.Module):
    def __init__(self, in_dim=64, h1=128, h2=128, h3=64, h4=32, dropout=0.2):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2),     nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h2, h3),     nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h3, h4),     nn.ReLU(), nn.Dropout(dropout),
        )
        self.out_dsl = nn.Linear(h4, 5)  # ΔSL(5)
        self.out_dll = nn.Linear(h4, 1)  # DLL 보조헤드(직접 회귀)

        # 명시적 가중치 초기화: Xavier (Glorot) 초기화로 seed 민감도 완화
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.backbone(x)
        dsl = self.out_dsl(h)                   # (B,5)
        dll_aux = self.out_dll(h)               # (B,1)
        return dsl, dll_aux

# -------------------
# 손실 함수들
# -------------------
import math, torch.nn.functional as F

def log_cosh_loss(pred, target):
    x = pred - target
    return torch.mean(x + F.softplus(-2.0*x) - math.log(2.0))

# -------------------
# 메인 루프
# -------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', type=str, required=True)
    ap.add_argument('--gt',  type=str, required=True)
    ap.add_argument('--replicate_globals', action='store_true', help='전역변수 5배 복제(→80차원). 미지정시 64차원')

    # 모델/학습
    ap.add_argument('--epochs', type=int, default=1000)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--alpha', type=float, default=1e-4)   # weight decay
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--hidden1', type=int, default=128)
    ap.add_argument('--hidden2', type=int, default=128)
    ap.add_argument('--hidden3', type=int, default=64)
    ap.add_argument('--hidden4', type=int, default=32)
    ap.add_argument('--seed', type=int, default=42)

    # 손실 가중
    ap.add_argument('--wSL', type=float, default=0.65)
    ap.add_argument('--wLL', type=float, default=0.30)
    ap.add_argument('--wC',  type=float, default=0.05)
    ap.add_argument('--tv_w', type=float, default=0.00)
    ap.add_argument('--beta_sl', type=float, default=0.5)

    # 고정 λ (합=1 필요 없음, 각 λ ∈ [0,1])
    ap.add_argument('--lambdas', type=str,
                    default='0.133191,0.113565,0.197085,0.207712,0.348446',
                    help='레벨별 λ 가중. 예: "0.133191,0.113565,0.197085,0.207712,0.348446"')

    # 얼리스탑(검증 total loss 기준)
    ap.add_argument('--patience', type=int, default=50)
    ap.add_argument('--min_delta', type=float, default=0.1)

    # 로깅
    ap.add_argument('--log_csv', type=str, default='runs_combined.csv')

    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # λ 파싱 (합=1 정규화 없음, 각 값 [0,1]로 clamp)
    lam_vals = [float(x) for x in args.lambdas.split(',')]
    assert len(lam_vals) == LEVELS, f"--lambdas 는 {LEVELS}개여야 합니다."
    lam_vals = np.clip(np.array(lam_vals, dtype=np.float32), 0.0, 1.0)
    lambdas = torch.tensor(lam_vals, device=device, dtype=torch.float32)  # (5,)

    ds = SpineDataset(args.raw, args.gt, replicate_globals=args.replicate_globals, zscore=True)
    if len(ds) == 0:
        print("데이터가 없습니다."); return

    tr_idx, va_idx, te_idx = split_indices(len(ds), seed=args.seed)
    tr_set, va_set, te_set = Subset(ds, tr_idx), Subset(ds, va_idx), Subset(ds, te_idx)
    
    # DataLoader worker 시드 고정 함수
    def worker_init_fn(worker_id):
        worker_seed = args.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    tr = DataLoader(tr_set, batch_size=args.batch_size, shuffle=True, 
                    worker_init_fn=worker_init_fn, generator=torch.Generator().manual_seed(args.seed))
    va = DataLoader(va_set, batch_size=args.batch_size, shuffle=False,
                    worker_init_fn=worker_init_fn)
    te = DataLoader(te_set, batch_size=args.batch_size, shuffle=False,
                    worker_init_fn=worker_init_fn)

    in_dim = ds.in_dim
    model = SharedMLP(in_dim=in_dim, h1=args.hidden1, h2=args.hidden2,
                      h3=args.hidden3, h4=args.hidden4, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.alpha)
    huber_sl = nn.SmoothL1Loss(beta=args.beta_sl, reduction='none')  # (B,5) 반환
    mse = nn.MSELoss()

    def run_epoch(loader, train=True):
        model.train(train)
        total, n = 0.0, 0
        pred_dll_all, gt_dll_all = [], []
        pred_dsl_all, gt_dsl_all = [], []

        for xb, y_sl_pre, y_sl_post, y_ll_pre, y_ll_post in loader:
            xb = xb.to(device).float()
            y_sl_pre  = y_sl_pre.to(device).float()
            y_sl_post = y_sl_post.to(device).float()
            y_ll_pre  = y_ll_pre.to(device).float().view(-1,1)
            y_ll_post = y_ll_post.to(device).float().view(-1,1)

            if train: opt.zero_grad(set_to_none=True)

            dsl_pred, dll_aux = model(xb)                # (B,5), (B,1)
            dll_sum = dsl_pred.sum(dim=1, keepdim=True)  # 합 방식

            # post 예측
            sl_post_pred = dsl_pred + y_sl_pre
            ll_post_sum  = dll_sum  + y_ll_pre
            ll_post_aux  = dll_aux  + y_ll_pre

            # -------- L_SL (고정 λ 가중; 합=1 필요 없음) --------
            per_el = huber_sl(sl_post_pred, y_sl_post)   # (B,5)
            per_level = torch.mean(per_el, dim=0)        # (5,)
            L_SL = torch.sum(lambdas * per_level)        # λ 가중 합

            # -------- L_LL / L_cons / L_tv --------
            L_LL   = log_cosh_loss(ll_post_sum, y_ll_post)     # 본판: 합 방식 평가
            L_cons = mse(dll_aux, dll_sum.detach())            # soft consistency
            L_tv   = torch.mean((dsl_pred[:,1:] - dsl_pred[:,:-1])**2) if args.tv_w>0 else dsl_pred.new_tensor(0.0)

            loss = args.wSL*L_SL + args.wLL*L_LL + args.wC*L_cons + args.tv_w*L_tv

            if not torch.isfinite(loss):
                # 문제 배치 스킵
                continue

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

            bs = xb.size(0)
            total += float(loss.item()) * bs
            n += bs

            # 지표 수집(검증/테스트용)
            pred_dll_all.append(ll_post_sum.detach().cpu())
            gt_dll_all.append(y_ll_post.detach().cpu())
            pred_dsl_all.append(sl_post_pred.detach().cpu())
            gt_dsl_all.append(y_sl_post.detach().cpu())

        # 집계
        import torch as _t
        pred_dll_all = _t.cat(pred_dll_all) if pred_dll_all else _t.zeros(0,1)
        gt_dll_all   = _t.cat(gt_dll_all)   if gt_dll_all   else _t.zeros(0,1)
        pred_dsl_all = _t.cat(pred_dsl_all) if pred_dsl_all else _t.zeros(0,LEVELS)
        gt_dsl_all   = _t.cat(gt_dsl_all)   if gt_dsl_all   else _t.zeros(0,LEVELS)

        avg_loss = total / max(n,1)
        rmse_dll_val = rmse(pred_dll_all, gt_dll_all)
        rmse_dsl_val = rmse(pred_dsl_all, gt_dsl_all)
        per_level_rmse = [rmse(pred_dsl_all[:,i], gt_dsl_all[:,i]) for i in range(LEVELS)]
        return avg_loss, rmse_dll_val, rmse_dsl_val, per_level_rmse

    # 조기종료/체크포인트: 검증 total loss 기준
    best_val_loss, best_epoch = float('inf'), -1
    patience_counter = 0
    ckpt_path = "checkpoint_best_by_val_loss.pt"

    for ep in range(1, args.epochs+1):
        tr_loss, tr_rmse_dll, tr_rmse_dsl, _ = run_epoch(tr, train=True)
        va_loss, va_rmse_dll, va_rmse_dsl, va_per = run_epoch(va, train=False)

        improved = va_loss < (best_val_loss - args.min_delta)
        if improved:
            best_val_loss = va_loss; best_epoch = ep; patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1

        print(f"[{ep:03d}] train_loss {tr_loss:.4f} | val_loss {va_loss:.4f} "
              f"| val_rmse_dll {va_rmse_dll:.3f} | val_rmse_dsl {va_rmse_dsl:.3f} "
              f"| λ {', '.join(f'{v:.3f}' for v in lambdas.tolist())} "
              f"| best_ep {best_epoch} ({best_val_loss:.4f})")

        if patience_counter >= args.patience:
            print(f"Early stop: {args.patience} epochs without val-loss improvement ≥ {args.min_delta}.")
            break

    # 테스트
    saved_lambdas = lambdas.detach().cpu().numpy()  # ← 기본값 미리 설정
    if Path(ckpt_path).exists():
        device_tgt = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(ckpt_path, map_location=device_tgt))
        
    te_loss, te_rmse_dll, te_rmse_dsl, te_per = run_epoch(te, train=False)
    print("\n[TEST] total_loss {:.4f} | RMSE_DLL {:.3f} | RMSE_DSL {:.3f} | per-level {}"
          .format(te_loss, te_rmse_dll, te_rmse_dsl, ", ".join(f"{v:.3f}" for v in te_per)))
    print("[INFO] Lambdas used:", ", ".join(f"{v:.6f}" for v in saved_lambdas))

    header = [
        "ts",
        "raw", "gt",
        "replicate_globals",
        "lr", "alpha", "dropout", "batch",
        "wSL", "wLL", "wC", "tv_w", "beta_sl",
        "lambdas",
        "best_epoch", "best_val_loss",
        "test_total_loss",
        "test_rmse_dll", "test_rmse_dsl",
        # per-level (있던 컬럼은 유지)
        "RMSE_SL1", "RMSE_SL2", "RMSE_SL3", "RMSE_SL4", "RMSE_SL5",
        # 필요시 평균도 기록 (기존에 있었다면 유지)
        "RMSE_SL_ALL",
        "in_dim"
    ]

    # te_rmse_dsl_all은 (있다면) 그대로 사용, 없으면 np.mean(te_per)로 대체
    try:
        sl_all = float(te_rmse_dsl_all)
    except NameError:
        sl_all = float(np.mean(te_per))  # te_per: per-level RMSE list/array

    row = [
        time.strftime('%Y-%m-%d %H:%M:%S'),
        args.raw, args.gt,
        int(args.replicate_globals),
        args.lr, args.alpha, args.dropout, args.batch_size,
        args.wSL, args.wLL, args.wC, args.tv_w, args.beta_sl,
        args.lambdas,
        best_epoch, best_val_loss,
        te_loss,
        te_rmse_dll, te_rmse_dsl,
        *te_per,                      # per-level 5개
        sl_all,
        ds.in_dim
    ]

    out_csv = Path(args.log_csv)
    new_file = (not out_csv.exists()) or (out_csv.stat().st_size == 0)

    with open(out_csv, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if new_file: w.writerow(header)
        w.writerow(row)

    print(f"[INFO] Saved summary to {out_csv}")

if __name__ == "__main__":
    main()
