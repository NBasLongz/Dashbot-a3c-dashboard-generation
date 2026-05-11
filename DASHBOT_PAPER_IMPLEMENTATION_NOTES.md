# DashBot Paper Implementation Notes

Paper: `2208.01232v3.pdf` - "DashBot: Insight-Driven Dashboard Generation Based on Deep Reinforcement Learning"

## Core Idea

DashBot frames analytical dashboard generation as a Markov Decision Process. An agent starts from a dashboard state, chooses a topic/key column, then repeatedly adds/removes charts or changes the key column until termination. The model is trained without a labeled dashboard dataset by using hand-designed rewards from visualization rules and statistical insight metrics.

## MDP Formulation

State:

- A dashboard state is a collection of valid charts: `{chart_j | j in [0, n]}`, where `n <= N`.
- Each chart is represented through Vega-Lite-like attributes.
- A dataset has at most 10 modeled columns; smaller datasets are zero-padded.

Actions:

- `change`: change key column/topic; all charts replace the old key column with the new one.
- `add`: add a new chart; requires selecting chart parameters.
- `remove`: remove an existing chart.
- `terminate`: stop exploration and finalize dashboard.

Important runtime limits from the paper:

- At most 50 exploration steps per dashboard.
- Initial generation quota: `n = 1000` agent steps.
- Online recommendation after user edit: `k = 200` steps.
- Training: 500,000 steps across 27 Vega datasets.

## Chart Parameter Space

The implementation focuses on:

- Mark types: `bar`, `line`, `point`, `boxplot`.
- Visual channels: `x`, `y`, `color`.
- Vega-Lite-style encodings.
- Column types: quantitative `Q`, temporal `T`, nominal `N`.

For `add`, the sequential prediction should produce parameters like:

- mark type
- key column handling
- explanation column / field
- aggregation per field
- color field / color usage

The exact parameter list is partly shown in the architecture figure rather than fully enumerated in text, so this must be reconstructed pragmatically from Vega-Lite constraints.

## Rewards

Immediate reward:

```text
r_i = f(s_i, a_i) = cr_i - cr_{i-1}
```

Dashboard combined reward:

```text
cr_i = w1 * sum(diversity rewards) + w2 * parsimony reward + w3 * sum(insight rewards)
```

Paper weights:

- `w1 = 0.33`
- `w2 = 0.33`
- `w3 = 0.1`

### Presentation Rewards

Diversity reward:

```text
dr = 1 - exp(-alpha * c_used / c_total)
```

Used twice:

- chart type diversity
- visualized column diversity

Parsimony reward:

```text
if n in [0, n_best]:
    pr = sin((pi / 2) * n / n_best)
else:
    pr = sin((pi / 2) * (1 + (n - n_best) / (n_max - n_best)))
```

Paper motivation:

- Most dashboards contain 3-6 views.
- Generated dashboards average 5.42 charts and 2.81 chart types.

Constants not fully specified in text:

- `alpha`
- `n_best`
- `n_max`
- insight thresholds, e.g. correlation threshold

Reasonable reproduction defaults:

- `n_best = 4`
- `n_max = 8`
- `alpha = 3`
- correlation threshold `abs(r) >= 0.5` or `0.6`
- `top_k = 5`

These defaults should be marked as reproduction assumptions unless source code is found.

### Insight Rewards

Reward value by insight arity:

- single-column insight: `1`
- double-column insight: `2`
- multiple-column insight: `3`

Insight definitions:

- `distribution`: `A in Q`; visualize `A` with histogram using bin count.
- `trend`: `A in Q`, `B in T`; visualize `A` across `B` with line chart.
- `correlation`: `A in Q`, `B in Q`; line chart or scatterplot, correlation above threshold.
- `top/bottom k`: `A in N`, `B in Q`; visualize top or bottom k entities of `A` by `B`.
- `co-correlation`: `A, B, C in Q`; correlation insights exist for `(A, B)` and `(A, C)`.
- `comparison`: `A in N`, `B in Q`; top and bottom k insights exist for `A` and `B`.

## Feature Engineering

Column features are based on VizML-like handcrafted statistics:

- data type
- min/max and other numeric summaries
- cardinality
- skewness
- Gini impurity
- entropy
- column name/id features mentioned in the figure

Chart features:

- one-hot mark type
- one-hot channel usage: `x`, `y`, `color`
- field features for encoded/transformed fields
- field features should describe rendered/transformed data, not raw data only. Example: `mean(US Gross) grouped by Major Genre`.

Dashboard feature:

- pack chart features into a sequence
- append current key column features
- append all dataset column features
- pad dataset columns to max 10
- output shape: `e_i in R^{n x l}`, where `n` is current chart count and `l` is chart feature length plus context

## Agent Network

Framework:

- A3C: asynchronous advantage actor-critic.
- Actor predicts action/parameter probabilities.
- Critic estimates expected return `v(s_i)`.

Loss:

```text
L(s_i, p_i) = (R - v(s_i))^2 - log(p_i) * A(s_i) - H(p_i)
```

Architecture:

- Input dashboard feature sequence.
- Bi-LSTM over chart features to model chart relationships.
- Randomly shuffle chart order during training because dashboard charts are a set, not an ordered sequence.
- Fuse Bi-LSTM output into one shared dashboard embedding.
- FC head for state value.
- Sequential classification blocks for action and parameters:
  - each block predicts one token/parameter
  - intermediate embedding from previous block is concatenated/fused with shared embedding
  - next block conditions on previous predictions

This sequential parameter prediction is important. The ablation without it converged lower and slower.

## Constrained Sampling

Constrained sampling is mandatory for a faithful implementation.

Apply masks before softmax so entropy remains valid for backprop.

The masks enforce:

- no non-existing columns
- only activate parameter branches relevant to the selected action
- prevent invalid Vega-Lite configurations
- prevent selecting the key column as an explanation column when inappropriate
- restrict aggregations by data type, e.g. disable `mean`/`max` for nominal explanation columns
- restrict mark/channel combinations according to visualization effectiveness rules

The paper reports that replacing constrained sampling with invalid-chart penalties is unstable.

## Interface Behavior

DashBot UI contains:

- table view: upload CSV, show columns and types, inspect raw data
- topic list: dashboards grouped by key column/topic and sorted by return
- chart editor: edit chart parameters, add chart, delete chart
- canvas view: display generated charts
- recommendation view: show charts recommended after user edits

Layout is rule-based, not the main model contribution:

- aggregate charts by mark type
- within same mark type, group charts with same insight type
- text summary/statistics go in the top row

## Evaluation Setup

Ablation baselines:

- DQN baseline with similar network, trained with temporal difference and replay memory
- `DashBot-ind.`: removes sequential dependency between classification blocks
- `DashBot-pen.`: removes constrained sampling and penalizes invalid charts

Hardware reported:

- Intel i7-8700K, 6 cores
- GTX 1080 Ti
- 32 GB memory

Datasets:

- 27 Vega datasets for training
- User study examples from Vega datasets such as cars, jobs, penguins, movies

## Practical Reimplementation Order

1. Build data profiling:
   - infer column type `Q/N/T`
   - compute column statistics and transformed field statistics

2. Build a Vega-Lite chart generator:
   - supported marks and channels only
   - grammar validation
   - deterministic conversion from internal chart object to Vega-Lite spec

3. Implement insight detectors:
   - distribution
   - trend
   - correlation
   - top/bottom k
   - co-correlation
   - comparison

4. Implement reward engine:
   - diversity reward
   - parsimony reward
   - insight reward
   - immediate reward as reward delta

5. Implement RL environment:
   - state = dashboard
   - actions = change/add/remove/terminate
   - action parameter execution
   - max steps and termination conditions

6. Implement constrained sampler:
   - action masks
   - column masks
   - mark masks
   - aggregate masks
   - channel masks

7. Implement A3C agent:
   - Bi-LSTM dashboard encoder
   - value head
   - sequential action/parameter heads
   - entropy regularization
   - asynchronous workers or a synchronous A2C-style approximation if exact async is too costly
   - initial assumptions: Adam optimizer, learning rate `1e-4`, Bi-LSTM hidden size `128` per direction, entropy coefficient `0.01`, value loss coefficient `0.5`

8. Implement generation loop:
   - run 1000 steps
   - collect terminal or high-return dashboards
   - group by key column
   - sort by return

9. Implement online recommendation:
   - start from edited dashboard state
   - run 200 steps
   - recommend high-reward additions/edits

## Known Gaps in the Paper

The paper does not provide enough detail to reproduce the system bit-for-bit:

- exact column feature vector schema
- exact parameter vocabulary for every classification block
- exact network dimensions
- optimizer and learning rate
- entropy coefficient / value loss coefficient
- insight thresholds
- `alpha`, `n_best`, and `n_max`
- full visualization constraint table
- source code or pretrained weights

For a course/project reproduction, document these as assumptions and keep the architecture faithful.
