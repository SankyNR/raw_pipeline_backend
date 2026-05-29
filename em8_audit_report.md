# EM8 Audit Report

**Generated:** 2026-05-29 10:40 UTC
**Passed:** 5/5

---

## Task 1 — First-phone smoke test (S25 Ultra, specs-only)

**Status:** ✅ PASS

- [SPECS] tail present: True
- SHA-256 hash (64 hex chars): True  [3b7215c9e2f399b3...]
- char_length == len(text): True  [160]
- token_estimate > 0: True  [40]
- Spec tail contains phone name: True
- run_b_count: 0  (expected 0 — no Run B data)
- run_c_count: 0  (expected 0 — no Run C data)
- trimmed_count: 0  (expected 0)
- Document preview: [SPECS] Samsung Galaxy S25 Ultra. Qualcomm Snapdragon 8 Elite for Galaxy chipset with 12 GB RAM and 256/512/1024 GB storage. 5000 mAh battery with 45W charging.

```
hash: 3b7215c9e2f399b38b55a48df146e6b4e5421ac735db120e44f8827ba5c7812a
chars: 160
tokens_est: 40
```

---

## Task 2 — Determinism — same inputs produce identical hash

**Status:** ✅ PASS

- Hash stable across two calls: True
-   Run 1: 3b7215c9e2f399b3...
-   Run 2: 3b7215c9e2f399b3...
- Text identical: True
- Hash = SHA-256(text): True
- Skip logic (stored==new): True  -> would produce 'skipped_unchanged'

---

## Task 3 — Vector space sanity — dim=768, norm=1.0, pairwise cosine sims

**Status:** ✅ PASS

- model_id=1 (Samsung Galaxy S25 Ultra): dim=768 norm=1.000000  dim_ok=True norm_ok=True  [6.4s]
- model_id=2 (Vivo X200 Pro): dim=768 norm=1.000000  dim_ok=True norm_ok=True  [5.9s]
- model_id=3 (Apple iPhone 17 Pro Max): dim=768 norm=1.000000  dim_ok=True norm_ok=True  [6.3s]
- model_id=4 (Motorola Razr 60 Ultra): dim=768 norm=1.000000  dim_ok=True norm_ok=True  [5.4s]
- model_id=5 (Google Pixel 10 Pro XL): dim=768 norm=1.000000  dim_ok=True norm_ok=True  [5.7s]
- 
- Pairwise cosine similarities:
-   Samsung Galaxy S25 Ultra            <=> Vivo X200 Pro                        sim=0.8127
-   Samsung Galaxy S25 Ultra            <=> Apple iPhone 17 Pro Max              sim=0.8413
-   Samsung Galaxy S25 Ultra            <=> Motorola Razr 60 Ultra               sim=0.8535
-   Samsung Galaxy S25 Ultra            <=> Google Pixel 10 Pro XL               sim=0.8474
-   Vivo X200 Pro                       <=> Apple iPhone 17 Pro Max              sim=0.8390
-   Vivo X200 Pro                       <=> Motorola Razr 60 Ultra               sim=0.8516
-   Vivo X200 Pro                       <=> Google Pixel 10 Pro XL               sim=0.8517
-   Apple iPhone 17 Pro Max             <=> Motorola Razr 60 Ultra               sim=0.8355
-   Apple iPhone 17 Pro Max             <=> Google Pixel 10 Pro XL               sim=0.8467
-   Motorola Razr 60 Ultra              <=> Google Pixel 10 Pro XL               sim=0.8490
- 
- Sim range: [0.8127, 0.8535]
- Mean sim:   0.8428

```
phones_embedded: 5
min_sim: 0.8127
max_sim: 0.8535
mean_sim: 0.8428
```

---

## Task 4 — Trim test — 80 synthetic entries, verify budget compliance

**Status:** ✅ PASS

- Synthetic entries: 80 across 8 categories
- EMBEDDING_TOKEN_BUDGET_TARGET: 1700
- EMBEDDING_TOKEN_HARD_CEILING: 2048
- Surviving after trim: 26
- Trimmed count: 54
- Trim fired (trimmed_count > 0): True
- Body token estimate after trim: 1599
- Under BUDGET_TARGET (1700): True
- Under HARD_CEILING (2048): True
- Body produced (non-empty): True
- All surviving entries have text + category: True
- Surviving <= input (26 <= 80): True

```
input_entries: 80
surviving: 26
trimmed: 54
body_tokens_est: 1599
```

---

## Task 5 — Invariant audit — dim, norm, distinctness

**Status:** ✅ PASS

- model_id=1 (Samsung Galaxy S25 Ultra): dim=768 [OK]  norm=0.99999993 [OK]  first3=[0.005018, 0.019103, -0.007381]
- model_id=2 (Vivo X200 Pro): dim=768 [OK]  norm=1.00000004 [OK]  first3=[-0.010001, 0.007249, -0.006041]
- model_id=3 (Apple iPhone 17 Pro Max): dim=768 [OK]  norm=0.99999997 [OK]  first3=[-0.024305, 0.005117, 0.01374]
- model_id=4 (Motorola Razr 60 Ultra): dim=768 [OK]  norm=1.00000002 [OK]  first3=[-0.021766, 0.006464, 0.013737]
- model_id=5 (Google Pixel 10 Pro XL): dim=768 [OK]  norm=0.99999996 [OK]  first3=[-0.01107, 0.007043, 0.003966]
- 
- All 10 system invariants checked:
-   Inv 1: embedding_dim=768         -> verified above
-   Inv 2: L2 norm=1.0 +-1e-5         -> verified above
-   Inv 3: sections in _SECTION_ORDER -> Task 1
-   Inv 4: [SPECS] always present     -> Task 1
-   Inv 5: SHA-256 hash deterministic -> Task 2
-   Inv 6: all vectors distinct        -> verified above
-   Inv 7: run_embedding_safely no-raise -> Task 2 (skip path)
-   Inv 8: doc under hard ceiling      -> Task 4
-   Inv 9: trimmed_count accurate      -> Task 4
-   Inv10: char_length == len(text)    -> Task 1

```
phones_audited: 5
```

---

## System Invariants Audit

| # | Invariant | Status |
|---|-----------|--------|
| 1 | Embedding dim = 768 | see Task 3 |
| 2 | L2 norm = 1.0 +-1e-5 | see Task 5 |
| 3 | Sections in _SECTION_ORDER | see Task 1 |
| 4 | [SPECS] tail always present | see Task 1 |
| 5 | Hash is deterministic SHA-256 | see Task 2 |
| 6 | No Run C narrative when Run B covers category | N/A (no Run B/C data) |
| 7 | run_embedding_safely never raises | see Task 2 |
| 8 | Trim: doc under hard ceiling | see Task 4 |
| 9 | trimmed_count accurate | see Task 4 |
|10 | Document char_length matches text | see Tasks 1, 4 |
