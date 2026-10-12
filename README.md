# Huong dan ban giao workspace

Tai lieu nay mo ta workspace `method` de agent tiep theo co the tim dung project, hieu cac goi Kaggle va tiep tuc cong viec. Workspace gom hai repository Git doc lap: `s2it/` va `tree/`. Thu muc `method/` hien khong phai repository Git.

## Muc luc nhanh

| Duong dan | Noi dung |
|---|---|
| `s2it/` | Cai dat S2IT cho Aspect Sentiment Quad Prediction, notebook Kaggle va goi offline |
| `tree/` | Cai dat Tree-CoT-RT, notebook Kaggle va goi offline |
| `.venv/` | Moi truong Python cap workspace; khong phai source code va khong nen dong goi |

Moi project co Git history rieng. Chay `git -C s2it status --short` hoac `git -C tree status --short` de xem thay doi cua repo tuong ung. Khong chay lenh Git tu thu muc `method/` de thay doi project.

## S2IT (`s2it/`)

| Tep/thu muc | Vai tro |
|---|---|
| `README.md` | Huong dan chi tiet S2IT, chuan bi du lieu/ngon ngu, train, predict va Kaggle |
| `2025.findings-naacl.379.pdf` | Bai bao tham khao cho phuong phap S2IT |
| `pyproject.toml` | Metadata, dependencies, lenh CLI va cau hinh pytest |
| `.gitignore` | Bo qua du lieu `asqp_data/`, virtualenv, cache, bytecode va egg-info |
| `s2it_kaggle.ipynb` | Notebook Kaggle chinh de chuan bi du lieu, huan luyen va danh gia |
| `src/asqp_s2it/__init__.py` | Khai bao package |
| `src/asqp_s2it/cli.py` | Cac lenh CLI train, predict, evaluate va parser |
| `src/asqp_s2it/data.py` | Doc/kiem tra du lieu JSONL va cac kieu du lieu ASQP |
| `src/asqp_s2it/language_prep.py` | Gan ma ngon ngu vao tung record JSONL |
| `src/asqp_s2it/metrics.py` | Tinh precision, recall va micro-F1 cho quadruple |
| `src/asqp_s2it/model.py` | Nap model/LoRA, sinh va parse du doan |
| `src/asqp_s2it/prompts.py` | Prompt cho trich cap aspect-opinion va phan loai |
| `src/asqp_s2it/syntax.py` | Stanza parser va bieu dien dependency relations/subgraph |
| `src/asqp_s2it/timing.py` | Tien ich do va ghi thoi gian |
| `src/asqp_s2it/training.py` | Tao du lieu huan luyen va chay cac giai doan S2IT |
| `scripts/prepare_offline_bundle.py` | Tao bundle phu thuoc/du lieu de chay Kaggle offline |
| `tests/` | Kiem thu data/ngon ngu, metrics, syntax va timing |
| `asqp_data/` | Du lieu train/dev/test cuc bo; bi ignore boi Git |
| `.venv/`, `.pytest_cache/`, `__pycache__/`, `*.egg-info/` | Moi truong va file tao tu dong; khong phai dau vao can commit |

### Cac goi Kaggle S2IT

- `kaggle_offline_upload.zip` la archive upload Kaggle hien co.
- `kaggle_offline_upload/` la thu muc da giai nen/dong goi, gom notebook, `project/` (ban source de notebook import) va `offline_bundle/` (JSONL, Stanza resources, wheelhouse).
- `offline_bundle/` o goc `s2it/` la nguon du lieu va dependencies offline duoc script tao; no khong nam trong danh sach Git tracked theo `.gitignore` hien tai.
- Notebook Kaggle va source trong `kaggle_offline_upload/project/` can duoc giu dong bo khi sua code. Sau khi sua source, tao lai goi va tai lai Kaggle Dataset; Kaggle Input cu co the van chua ban code moi.
- Archive chi dong goi source, data va dependencies phuc vu notebook. Model LLM va dau ra training khong nen mac dinh nam trong archive; model duoc gan nhu Kaggle Input rieng, dau ra nam trong `/kaggle/working/`.

## Tree-CoT-RT (`tree/`)

| Tep/thu muc | Vai tro |
|---|---|
| `README.md` | Huong dan su dung project Tree-CoT-RT |
| `2026.findings-acl.806.pdf` | Bai bao tham khao |
| `pyproject.toml`, `.gitignore` | Dependencies/cau hinh Python va quy tac bo qua Git |
| `tree_cot_rt_kaggle.ipynb` | Notebook Kaggle chinh |
| `src/tree_cot_rt/__init__.py` | Khai bao package |
| `src/tree_cot_rt/cli.py` | Lenh dong cho project |
| `src/tree_cot_rt/data.py` | Doc va chuan hoa du lieu |
| `src/tree_cot_rt/reasoning.py` | Tao/phan tich reasoning theo cau truc cay |
| `src/tree_cot_rt/rewards.py` | Tinh reward cho huan luyen |
| `src/tree_cot_rt/training.py` | Pipeline huan luyen va du doan |
| `tests/test_reasoning.py` | Kiem thu reasoning |
| `asqp_data/` | Du lieu train/dev/test cuc bo; bi ignore boi Git |
| `kaggle_offline_upload/` | Noi dung da giai nen de upload, gom notebook, README upload, project va offline bundle |
| `kaggle_offline_upload.zip` | Archive upload Kaggle |

Trong goi Tree-CoT-RT, `UPLOAD_README.txt` ghi ro can gan Kaggle Model rieng, bat GPU, khong can Internet; archive khong chua model weights.

## Vi sao dong goi ZIP?

ZIP la cach giao nhieu loai tai san thanh mot Kaggle Dataset de upload/gan vao notebook chi trong mot lan. Kaggle co the su dung notebook, source Python, du lieu JSONL va cac wheel Stanza trong mot input offline co cau truc on dinh. Trong archive, duong dan dung dau `/` de tuong thich voi Kaggle/Linux. Model lon duoc gan rieng qua Kaggle Model thay vi nhan ban trong archive.

Thu muc `kaggle_offline_upload/` giup kiem tra/chinh sua noi dung truoc khi tao archive; file ZIP la artifact upload da dong goi. Khi source thay doi, can tao lai ZIP tu ban thu muc hien hanh va cap nhat Kaggle Dataset. Khong sua truc tiep mot ban giai nen cu roi quen cap nhat archive.

## Tep khong dua len GitHub

Hien tai, trong moi repo, `git status` la nguon chinh xac nhat de biet tep nao chua tracked. `.gitignore` khong cho them mot so tep vao Git; cac goi Kaggle va archive co the hien la untracked cho den khi duoc chu dong commit.

| Nhom | Vi du | Ly do khong dua len GitHub |
|---|---|---|
| Du lieu | `s2it/asqp_data/`, `tree/asqp_data/`, JSONL trong offline bundle | Co the lon, co han che phan phoi/giay phep; quan ly va cap phat qua Kaggle Dataset hoac nguon du lieu duoc phep |
| Model/checkpoint | Model goc, LoRA adapter, weights | Rat lon; quan ly qua Kaggle Models hoac kho artifact phu hop |
| Dependencies offline | `wheelhouse/*.whl`, `stanza_resources/**/*.pt` | Binary lon, phu thuoc phien ban/nha cung cap; chi dua vao goi offline can upload |
| Goi upload | `kaggle_offline_upload/`, `kaggle_offline_upload.zip` | Artifact tao ra de phuc vu Kaggle; co the tai tao tu source va script dong goi |
| Moi truong va cache | `.venv/`, `.pytest_cache/`, `__pycache__/`, `*.pyc`, `*.egg-info/` | Rieng may, tao tu dong, khong can thiet cho nguoi khac clone project |
| Ket qua chay | Log, predictions, checkpoints, training output | Co the nang va tao lai; chi luu trong repo neu co muc dich phan tich/tai lap cu the |

Khong commit file chi vi no xuat hien trong workspace. Truoc khi commit, kiem tra noi dung, kich thuoc, quyen phan phoi du lieu/model va `git status --short`. README khong tu dong thay doi `.gitignore` hay upload bat ky tep nao.

## Trang thai lan chay S2IT gan nhat

Theo log duoc cung cap, ca ba giai doan fine-tuning hoan tat; adapter va tokenizer da luu vao `/kaggle/working/s2it_outputs/final`, metrics vao `final/training_metrics.json`, va thoi gian vao `training_time.csv`. Notebook sau do gap `JSONDecodeError` tai buoc inference/danh gia trong `predict()`, luc parse output classification cua model. Day la loi output JSON cua lan suy luan, khong phai loi huan luyen; can chay lai evaluation sau khi sua/bo sung xu ly output loi. Cac warning `temperature/top_p/top_k` bi bo qua khi `do_sample=False`, khong phai nguyen nhan traceback.

## Huong dan cho agent tiep theo

1. Xac dinh project nguoi dung dang noi toi (`s2it/` hay `tree/`) va doc README tuong ung cung README nay.
2. Kiem tra `git -C <project> status --short` truoc khi sua; khong ghi de thay doi dang co.
3. Neu thay doi source S2IT dung cho Kaggle, cap nhat ca `src/asqp_s2it/` va ban source dong goi trong `kaggle_offline_upload/project/src/asqp_s2it/`; kiem tra notebook import tu source nao.
4. Neu can tao lai goi offline, xem `scripts/prepare_offline_bundle.py`, `UPLOAD_README.txt` (Tree-CoT-RT) va README project. Xac nhan danh sach file trong ZIP truoc khi upload Kaggle.
5. Khong chay lai huan luyen S2IT mac dinh: lan gan nhat mat hon 8 gio va adapter da duoc luu tren Kaggle. Truoc tien danh gia lai tu adapter hien co neu no con truy cap duoc. Model/checkpoint nam ngoai repo may cuc bo cho den khi duoc tai ve.
6. Chi chay kiem thu/lenh dai khi yeu cau cong viec can; khong co lenh training nao duoc tu dong thuc hien boi README nay.

## Trang thai Git luc tao tai lieu

Luc kiem tra, repo S2IT co `s2it_kaggle.ipynb` thay doi chua commit; `kaggle_offline_upload/`, `offline_bundle/` va `kaggle_offline_upload.zip` chua tracked. Repo Tree-CoT-RT co thay doi chua commit tai `src/tree_cot_rt/reasoning.py`, `src/tree_cot_rt/training.py` va `tree_cot_rt_kaggle.ipynb`; `kaggle_offline_upload/` va `kaggle_offline_upload.zip` chua tracked. Hay xem lai `git status --short` truoc khi sua/commit vi trang thai co the doi. Cac thay doi nay la cua nguoi dung va phai duoc bao toan.
