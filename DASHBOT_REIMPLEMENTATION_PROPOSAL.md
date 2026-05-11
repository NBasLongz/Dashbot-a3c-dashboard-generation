# De Cuong Tai Hien DashBot

## 1. Muc Tieu

Do an tai hien paper **DashBot: Insight-Driven Dashboard Generation Based on Deep Reinforcement Learning** voi trong tam la xay dung mot he thong sinh dashboard phan tich du lieu dua tren Deep Reinforcement Learning.

He thong can tu dong:

- doc du lieu bang tu file CSV;
- suy luan kieu cot va dac trung thong ke;
- sinh cac bieu do hop le theo grammar gan voi Vega-Lite;
- danh gia dashboard bang presentation rewards va insight rewards;
- huan luyen agent RL de chon topic/key column va cau hinh dashboard;
- ho tro goi y dashboard/chart sau khi nguoi dung chinh sua.

## 2. Linh Hon Ky Thuat Cua Paper

DashBot khong hoc tu tap dashboard gan nhan. Thay vao do, paper bien qua trinh tao dashboard thanh mot **Markov Decision Process**:

- **State**: dashboard hien tai, gom tap cac chart hop le.
- **Action**: `change`, `add`, `remove`, `terminate`.
- **Reward**: tinh tu chat luong trinh bay va insight thong ke.
- **Policy**: A3C agent hoc cach sinh chuoi hanh dong de toi da hoa reward.

Diem quan trong nhat cua paper la **constrained sampling**. Agent khong duoc tu do sinh moi tham so chart, ma moi prediction se bi mask boi cac rang buoc visualization grammar va data type truoc khi softmax. Co che nay giup tranh chart vo nghia hoac render loi, va tot hon cach chi phat diem am cho chart sai.

## 3. MDP Formulation

### State Space

Mot state la mot dashboard:

```text
s_i = { chart_j | j in [0, n], n <= N }
```

Moi chart duoc ma hoa bang cac thuoc tinh:

- mark type;
- x/y/color channel;
- field duoc chon;
- aggregate/bin neu co;
- insight type neu phat hien duoc.

### Action Space

```text
A = { change, add, remove, terminate }
```

Y nghia:

- `change`: doi key column/topic cua dashboard.
- `add`: them chart moi, can sinh tiep cac tham so chart.
- `remove`: xoa mot chart hien co.
- `terminate`: ket thuc episode va chap nhan dashboard.

Gioi han theo paper:

- toi da 50 buoc exploration cho moi dashboard;
- sinh dashboard ban dau voi quota 1000 buoc;
- online recommendation sau edit voi quota 200 buoc.

## 4. Reward Design

Immediate reward:

```text
r_i = cr_i - cr_{i-1}
```

Combined dashboard reward:

```text
cr_i = w1 * diversity + w2 * parsimony + w3 * insight
```

Gia tri theo paper:

```text
w1 = 0.33
w2 = 0.33
w3 = 0.1
```

### 4.1 Diversity Reward

```text
dr = 1 - exp(-alpha * c_used / c_total)
```

Dung cho:

- do da dang loai chart;
- do da dang cot duoc visualize.

Assumption de xuat:

```text
alpha = 2.0 hoac 3.0
```

Ly do: paper chi dung 4 mark type co ban (`bar`, `line`, `point`, `boxplot`). Voi `alpha = 2` hoac `3`, khi dashboard da dung khoang 3/4 loai chart thi reward bat dau bao hoa, dung tinh than diminishing returns.

### 4.2 Parsimony Reward

Paper cho biet dashboard tot thuong co 3-6 view va rat it khi vuot qua 8.

Assumption de xuat:

```text
n_best = 4
n_max = 8
```

Cong thuc:

```text
if n <= n_best:
    pr = sin((pi / 2) * n / n_best)
else:
    pr = sin((pi / 2) * (1 + (n - n_best) / (n_max - n_best)))
```

Y nghia: reward tang den khi dashboard co khoang 4 chart, sau do giam dan neu dashboard qua day.

### 4.3 Insight Reward

Gia tri insight:

- single-column insight: `1`;
- double-column insight: `2`;
- multiple-column insight: `3`.

Insight can cai dat:

| Insight | Dieu kien |
|---|---|
| distribution | `A in Q`, histogram/bin count |
| trend | `A in Q`, `B in T`, line chart |
| correlation | `A in Q`, `B in Q`, Pearson correlation vuot nguong |
| top/bottom k | `A in N`, `B in Q`, group by `A`, sort theo `B` |
| co-correlation | `A, B, C in Q`, ton tai correlation `(A,B)` va `(A,C)` |
| comparison | `A in N`, `B in Q`, ton tai ca top k va bottom k |

Assumption de xuat:

```text
correlation_threshold = 0.5 hoac 0.6
top_k = 5 hoac 10
```

Trong ban dau nen dung:

```text
correlation_threshold = 0.5
top_k = 5
```

Ly do: nguong 0.5 bat duoc tuong quan trung binh-kha, phu hop cho dashboard recommendation. `top_k = 5` giup chart gon va de doc.

## 5. Feature Engineering

### Column Features

Moi cot can co vector dac trung:

- type id: quantitative, nominal, temporal;
- missing ratio;
- cardinality;
- min, max, mean, std voi cot quantitative;
- skewness;
- entropy;
- Gini impurity;
- normalized uniqueness;
- optional: name/id embedding don gian.

### Chart Features

Moi chart can ma hoa:

- one-hot mark type;
- one-hot channel usage;
- field feature cua x/y/color;
- aggregate/bin flag;
- insight flags.

Luu y cua paper: field feature nen tinh tren du lieu sau bien doi, vi du `mean(US Gross) grouped by Major Genre`, khong chi copy feature raw column.

### Dashboard Features

Dashboard feature la sequence cac chart feature, co them context:

- current key column feature;
- all dataset column features;
- zero padding den toi da 10 cot;
- random shuffle chart order trong training de giam phu thuoc thu tu chart.

## 6. Agent Architecture

Paper dung A3C:

- actor: du doan action va parameter probabilities;
- critic: uoc luong state value.

Loss:

```text
L = (R - v(s_i))^2 - log(p_i) * A(s_i) - H(p_i)
```

Kien truc de tai hien:

- Bi-LSTM encoder tren sequence chart features;
- fuse hidden states thanh shared dashboard embedding;
- value head cho critic;
- sequential classification heads cho actor:
  - action head;
  - key column head;
  - mark head;
  - field/channel/aggregate heads;
  - remove-index head.

Assumption hyperparameters:

```text
optimizer = Adam
learning_rate = 1e-4 hoac 5e-4
bilstm_hidden_size = 128 hoac 256 moi chieu
entropy_coef = 0.01 hoac 0.05
value_loss_coef = 0.5
gamma = 1.0
max_episode_steps = 50
training_steps = 500000
```

Gia tri nen chon cho ban dau:

```text
learning_rate = 1e-4
bilstm_hidden_size = 128
entropy_coef = 0.01
```

Ly do: cac gia tri nay on dinh hon cho reproduction nho, tranh policy dao dong manh khi reward con dang duoc tinh chinh.

## 7. Constrained Sampling

Mask phai ap dung **truoc softmax**.

Can co cac nhom constraint:

- action mask: khong cho `remove` neu dashboard rong, khong cho `add` neu dat `n_max`;
- column mask: khong chon cot khong ton tai, khong chon key column lam explanation column neu khong hop le;
- mark mask: gioi han mark theo data type va channel;
- aggregate mask: khong `mean/max` tren nominal field;
- channel mask: tranh encoding vo nghia, vi du nominal vao size/continuous channel neu khong ho tro;
- grammar mask: chi sinh Vega-Lite spec render duoc.

Diem nay phai duoc uu tien cao hon model architecture, vi paper cho thay penalty cho invalid chart khong on dinh bang constrained sampling.

## 8. Lo Trinh Trien Khai

### Phase 1: Core Data va Reward

Muc tieu: co reward engine dung truoc khi huan luyen RL.

Can code:

- `DataProfiler`;
- `ChartSpec`;
- `InsightDetector`;
- `RewardEngine`;
- unit tests cho correlation, top/bottom k, parsimony, diversity.

### Phase 2: Dashboard Environment

Can code:

- `DashBotEnv.reset()`;
- `DashBotEnv.step(action, params)`;
- action execution;
- terminal condition;
- reward delta.

### Phase 3: Rule-Based/Random Baseline

Truoc khi A3C, can co baseline de kiem tra reward:

- random agent co constrained sampling;
- greedy agent chon action/param co reward cao nhat trong mot tap ung vien nho.

Neu reward engine dung, baseline da sinh duoc dashboard kha hop ly.

### Phase 4: A3C/A2C Agent

Can code:

- PyTorch feature encoder;
- Bi-LSTM dashboard encoder;
- actor sequential heads;
- critic value head;
- masked categorical sampling;
- rollout buffer;
- training loop.

Neu async A3C qua nang, co the dung synchronous A2C truoc va ghi ro la approximation cua A3C.

### Phase 5: UI Integration

Hien tai repo co `index.html` frontend demo. Sau khi backend on dinh, co the tich hop:

- upload CSV;
- goi backend sinh dashboard;
- render chart bang Chart.js hoac Vega-Lite;
- hien topic/key column list;
- hien reward va insight badges;
- chart editor goi online recommendation.

## 9. Known Gaps Va Assumptions

Paper khong cong bo day du:

- exact feature vector schema;
- exact action parameter vocabulary;
- optimizer;
- learning rate;
- network hidden dimensions;
- entropy/value loss coefficients;
- insight thresholds;
- `alpha`, `n_best`, `n_max`;
- full visualization constraint table;
- source code/pretrained weights.

Do do, do an se cong bo assumption nhu sau:

| Thanh phan | Assumption ban dau |
|---|---|
| `alpha` | `3.0` |
| `n_best` | `4` |
| `n_max` | `8` |
| correlation threshold | `0.5` |
| top/bottom k | `5` |
| optimizer | Adam |
| learning rate | `1e-4` |
| Bi-LSTM hidden size | `128` moi chieu |
| entropy coefficient | `0.01` |
| value loss coefficient | `0.5` |
| gamma | `1.0` |
| max columns | `10` |
| max episode steps | `50` |
| training steps | `500000` |

## 10. San Pham Du Kien

Ket qua mong muon:

- mot DashBot prototype co the doc CSV va sinh dashboard;
- reward engine co test rieng;
- baseline random/greedy de so sanh;
- RL agent co training log;
- report giai thich ro cac assumption va vi sao chung hop ly;
- demo UI hien dashboard, insight va topic list.

