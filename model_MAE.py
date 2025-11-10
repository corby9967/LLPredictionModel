#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_mae.py - SL_post 및 LL_post 예측 (Delta SL(5) 예측 기반 합-방식), DLL 보조헤드 (MAE / L1Loss 버전)
- 지표: MAE_LL_post, MAE_SL_post (전체 및 레벨별, 평균)로 명확히 수정됨.
- 입력: (5레벨x12) + 글로벌4 = 64차원 [기본]. --replicate_globals 로 80차원 전환 가능
- 손실:
    L_SL    = L1Loss on SL_post  (레벨별 lambda 가중 적용)
    L_LL    = L1Loss on LL_post (합-방식 dll_sum 사용)
    L_cons= L1Loss(dll_aux, dll_sum.detach())  # 합-보조 일관성 soft 제약
    L_tv    = 인접 레벨 Delta SL 스무딩(옵션; tv_w가 0이면 꺼짐)
    total = wSL*L_SL + wLL*L_LL + wC*L_cons + tv_w*L_tv
- 체크포인트/조기종료: **검증 total loss 기준**
- CSV: **mae_ll_post**, **mae_sl_post_all**, **mae_sl_post_all_avg**, per-level mae, best_epoch, lambdas 기록
"""

import json, argparse, random, csv, time, os
from pathlib import Path
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import math, torch.nn.functional as F

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
    
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG'):
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:
            print(f"Warning: unable to enable torch.use_deterministic_algorithms: {e}")
    else:
        pass

def split_indices(n, seed=None, ratios=(0.7, 0.15, 0.15)):
    ids = list(range(n))
    rnd = random.Random(seed); rnd.shuffle(ids)
    n_tr = int(n * ratios[0]); n_va = int(n * ratios[1])
    tr = ids[:n_tr]; va = ids[n_tr:n_tr+n_va]; te = ids[n_tr+n_va:]
    return tr, va, te

# 💡 RMSE 대신 MAE 함수 정의
def mae(a: torch.Tensor, b: torch.Tensor) -> float:
    # a, b 모두 0차원 이상의 텐서여야 합니다.
    if a.numel() == 0: return 0.0
    return float(torch.mean(torch.abs(a - b)).item())

# -------------------
# 데이터셋 (64/80 차원)
# (이 부분은 동일하므로 생략 가능하나, 완전성을 위해 포함)
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
        self.out_dsl = nn.Linear(h4, 5)  # Delta SL(5)
        self.out_dll = nn.Linear(h4, 1)  # Delta LL 보조헤드(직접 회귀)

        # 명시적 가중치 초기화: Xavier (Glorot) 초기화로 seed 민감도 완화
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.backbone(x)
        dsl = self.out_dsl(h)              # (B,5)
        dll_aux = self.out_dll(h)          # (B,1)
        return dsl, dll_aux

# -------------------
# 손실 함수들 (Log-Cosh 대신 L1Loss 사용)
# -------------------
# 💡 log_cosh_loss 제거. nn.L1Loss를 사용합니다.

# -------------------
# 메인 루프
# -------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', type=str, required=True)
    ap.add_argument('--gt',  type=str, required=True)
    ap.add_argument('--replicate_globals', action='store_true', help='전역변수 5배 복제(->80차원). 미지정시 64차원')

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
    # 💡 L1Loss (MAE)를 사용하므로 beta_sl은 무시되거나 제거될 수 있습니다. 여기서는 제거합니다.
    # ap.add_argument('--beta_sl', type=float, default=0.5)

    # 고정 lambda (합=1 필요 없음, 각 lambda ∈ [0,1])
    ap.add_argument('--lambdas', type=str,
                     default='0.133191,0.113565,0.197085,0.207712,0.348446',
                     help='레벨별 lambda 가중. 예: "0.133191,0.113565,0.197085,0.207712,0.348446"')

    # 얼리스탑(검증 total loss 기준)
    ap.add_argument('--patience', type=int, default=50)
    ap.add_argument('--min_delta', type=float, default=0.1)

    # 로깅
    ap.add_argument('--log_csv', type=str, default='runs_combined_mae.csv')

    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # lambda 파싱 (합=1 정규화 없음, 각 값 [0,1]로 clamp)
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
    
    # 💡 L1Loss (MAE)로 변경
    l1_sl = nn.L1Loss(reduction='none')  # (B,5) 반환
    l1_ll = nn.L1Loss(reduction='mean')  # 스칼라 반환 (LL_post용)
    l1_cons = nn.L1Loss(reduction='mean') # 스칼라 반환 (일관성 제약용)
    
    def run_epoch(loader, train=True):
        model.train(train)
        total, n = 0.0, 0
        
        # LL_post 및 SL_post 평가를 위한 집계 변수
        pred_ll_post_all, gt_ll_post_all = [], []
        pred_sl_post_all, gt_sl_post_all = [], []

        for xb, y_sl_pre, y_sl_post, y_ll_pre, y_ll_post in loader:
            xb = xb.to(device).float()
            y_sl_pre  = y_sl_pre.to(device).float()
            y_sl_post = y_sl_post.to(device).float()
            y_ll_pre  = y_ll_pre.to(device).float().view(-1,1)
            y_ll_post = y_ll_post.to(device).float().view(-1,1)

            if train: opt.zero_grad(set_to_none=True)

            dsl_pred, dll_aux = model(xb)              # (B,5), (B,1)
            dll_sum = dsl_pred.sum(dim=1, keepdim=True)  # 합 방식

            # post 예측 (LL과 SL 모두)
            sl_post_pred = dsl_pred + y_sl_pre
            ll_post_sum  = dll_sum  + y_ll_pre

            # -------- L_SL (고정 lambda 가중; SL_post 기준) --------
            # 💡 Huber 대신 L1Loss 사용
            per_el = l1_sl(sl_post_pred, y_sl_post)  # (B,5)
            per_level = torch.mean(per_el, dim=0)        # (5,)
            L_SL = torch.sum(lambdas * per_level)        # lambda 가중 합

            # -------- L_LL / L_cons / L_tv --------
            # 💡 Log-Cosh 대신 L1Loss 사용
            L_LL   = l1_ll(ll_post_sum, y_ll_post)       # 본판: LL_post 합 방식 평가 (L1Loss)
            # 💡 MSE 대신 L1Loss 사용
            L_cons = l1_cons(dll_aux, dll_sum.detach())  # soft consistency (L1Loss)
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

            # 지표 수집 (LL_post, SL_post 기준)
            pred_ll_post_all.append(ll_post_sum.detach().cpu())
            gt_ll_post_all.append(y_ll_post.detach().cpu())
            pred_sl_post_all.append(sl_post_pred.detach().cpu())
            gt_sl_post_all.append(y_sl_post.detach().cpu())

        # 집계
        import torch as _t
        pred_ll_post_all = _t.cat(pred_ll_post_all) if pred_ll_post_all else _t.zeros(0,1)
        gt_ll_post_all   = _t.cat(gt_ll_post_all)   if gt_ll_post_all   else _t.zeros(0,1)
        pred_sl_post_all = _t.cat(pred_sl_post_all) if pred_sl_post_all else _t.zeros(0,LEVELS)
        gt_sl_post_all   = _t.cat(gt_sl_post_all)   if gt_sl_post_all   else _t.zeros(0,LEVELS)

        avg_loss = total / max(n,1)
        
        # 💡 지표 이름 변경: mae_ll_post, mae_sl_post_all, per_level_mae, mae_sl_post_all_avg
        mae_ll_post_val = mae(pred_ll_post_all, gt_ll_post_all)
        # 1. 전체 SL_post 요소에 대한 MAE (평탄화 방식)
        mae_sl_post_all_val = mae(pred_sl_post_all, gt_sl_post_all)
        # 2. 레벨별 MAE (5개)
        per_level_mae = [mae(pred_sl_post_all[:,i], gt_sl_post_all[:,i]) for i in range(LEVELS)]
        # 3. 레벨별 MAE의 산술 평균
        mae_sl_post_all_avg = float(np.mean(per_level_mae))

        # 반환 값 이름 변경 반영
        return avg_loss, mae_ll_post_val, mae_sl_post_all_val, per_level_mae, mae_sl_post_all_avg

    # 조기종료/체크포인트: 검증 total loss 기준
    best_val_loss, best_epoch = float('inf'), -1
    patience_counter = 0
    ckpt_path = "checkpoint_best_by_val_loss_mae.pt"

    # va_per, va_mae_sl_post_all_avg 초기화
    va_per = [0.0] * LEVELS
    va_mae_sl_post_all_avg = 0.0

    for ep in range(1, args.epochs+1):
        # 반환 값 이름 변경 반영
        tr_loss, tr_mae_ll_post, tr_mae_sl_post_all, _, _ = run_epoch(tr, train=True)
        va_loss, va_mae_ll_post, va_mae_sl_post_all, va_per, va_mae_sl_post_all_avg = run_epoch(va, train=False)

        improved = va_loss < (best_val_loss - args.min_delta)
        if improved:
            best_val_loss = va_loss; best_epoch = ep; patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1

        # 출력 문구 수정: MAE_LL_post 및 MAE_SL_post로 변경
        print(f"[{ep:03d}] train_loss {tr_loss:.4f} | val_loss {va_loss:.4f} "
              f"| val_MAE_LL_post {va_mae_ll_post:.3f} | val_MAE_SL_post_ALL {va_mae_sl_post_all:.3f} "
              f"(Avg: {va_mae_sl_post_all_avg:.3f}) "
              f"| lambda {', '.join(f'{v:.3f}' for v in lambdas.tolist())} "
              f"| best_ep {best_epoch} ({best_val_loss:.4f})")

        if patience_counter >= args.patience:
            print(f"Early stop: {args.patience} epochs without val-loss improvement >= {args.min_delta}.")
            break

    # 테스트
    saved_lambdas = lambdas.detach().cpu().numpy()
    if Path(ckpt_path).exists():
        device_tgt = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(ckpt_path, map_location=device_tgt))
        
    # 테스트 결과 변수 이름 변경 반영
    te_loss, te_mae_ll_post, te_mae_sl_post_all, te_per, te_mae_sl_post_all_avg = run_epoch(te, train=False)
    
    # 테스트 결과 출력 문구 수정: MAE_LL_post 및 MAE_SL_post로 변경
    print("\n[TEST] total_loss {:.4f} | MAE_LL_post {:.3f} | MAE_SL_post_ALL {:.3f} | MAE_SL_post_ALL_Avg {:.3f} | per-level {}"
          .format(te_loss, te_mae_ll_post, te_mae_sl_post_all, te_mae_sl_post_all_avg, ", ".join(f"{v:.3f}" for v in te_per)))
    print("[INFO] Lambdas used:", ", ".join(f"{v:.6f}" for v in saved_lambdas))

    # CSV 헤더 수정: RMSE -> MAE로 변경
    header = [
        "ts",
        "raw", "gt",
        "replicate_globals",
        "lr", "alpha", "dropout", "batch",
        "wSL", "wLL", "wC", "tv_w",
        # "beta_sl", # L1Loss 사용으로 인해 제거
        "lambdas",
        "best_epoch", "best_val_loss",
        "test_total_loss",
        "test_mae_LL_post", 
        "test_mae_SL_post_ALL", 
        "test_mae_SL_post_ALL_Avg", 
        "MAE_SL1", "MAE_SL2", "MAE_SL3", "MAE_SL4", "MAE_SL5",
        "in_dim"
    ]

    # 데이터 행 수정: RMSE -> MAE로 변경
    row = [
        time.strftime('%Y-%m-%d %H:%M:%S'),
        args.raw, args.gt,
        int(args.replicate_globals),
        args.lr, args.alpha, args.dropout, args.batch_size,
        args.wSL, args.wLL, args.wC, args.tv_w,
        # args.beta_sl, # L1Loss 사용으로 인해 제거
        args.lambdas,
        best_epoch, best_val_loss,
        te_loss,
        te_mae_ll_post, 
        te_mae_sl_post_all, 
        te_mae_sl_post_all_avg, 
        *te_per, # per-level 5개
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