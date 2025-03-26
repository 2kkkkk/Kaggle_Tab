# Jane street

The competition dataset comprises a set of timeseries with 79 features and 9 responders, anonymized but representing real market data. The goal of the competition is to forecast one of these responders, i.e., `responder_6`, for up to six months in the future.

## My story

读了public notebook后，决定

1）用Public notebook现成代码训练xgb lgb cbt ，建立CV策略

2）use NN

3）Online learning

4）不用lag特征，因为看评论感觉lag特征没有用



**CV部分：**

看了别人的EDA，在date_id=676之前，每个date有849个time_id，而在date_id=676之后 ，每个date有968个time_id，所以我打算根据这个特点并结合模型训练的数据量建立CV，将数据分为3份：

1）date_id 0-500训练集，date_id 500-676验证集（因为private test的数据是未来6个月，所以验证集先选择120天左右的数据）

2）date_id 677-1100训练集，date_id 1101-1221验证集

3）date_id 1101-1577训练集，date_id 1578-1698验证集

分别用这三份数据训练了xgb,lgb,cbt，发现用数据集3）训练的模型得分最高（我分析，因为数据本身就是non-stationary的，所以不一定训练数据越多越好，越靠近当前date的数据越有用，所以要用online learning），而且多模型平均得分比单模型好，所以我决定：

1）加入lag特征（虽然作用不大，但是还是加上吧）

2）要预测未来6个月，只用offline model不行，要用online learning

3）看了public notebook，决定尝试FT-Transformer和tabm

4）模型ensemble可以提高得分且robust

5）试试post process

新的CV策略：用date_id=1100天到1668天做训练集，1668-1698天做验证集，训练 xgb,lgb,cbt，得到best n_iter。然后再用1100-1698所有数据训练，n_iter保持不变或者稍微多一些。



**Online Learning 部分：**

参考了public 的online learning notebook，Lightgbm的策略是每20天训练一次模型，训练集为当前date的前300天数据，n_iter固定为60，lr固定为0.1，特征用重要性top45的特征，不用全部的特征，不然会超出1分钟的时间限制。但是lgb的在线学习并没有提高public score。

NN的策略是每n天（n=1,n=5,n=20,都试了）用前n天的数据反向传播，更新参数，lr = 1e-4，epoch=1，但是没有提升，而且只要是lr稍微大一点，epoch稍微多一点，分数就会下降较多，看评论分析可能是用了batchnorm的原因？



**模型部分：**

尝试了NN模型中用前一天lag的agg 特征，但是得分下降了，所以不打算用了。

自己写了FT-Transformer的代码，但是提交后得分不高，放弃了。

public的tabm得分也不高。

看了public EDA，发现feature09 feature10 feature11这3个是类别特征，所以把public的NN代码改成加入category embedding的版本，但是得分没有提升，放弃了，改为把3个类别特征直接舍弃，score能提升0.0001。

public notebook中加了Ridge regression也能提升，我也打算在online learning部分加上，策略跟lgb一样，训练数据改为当前date前100天，不然会oom。

单个NN得分不如5-fold的nn ensemble得分高，所以打算用3fold nn ensemble。

最后的xgb打算用public的，因为得分比我的高，而且感觉也比较robust。

最后的版本：public xgb + 我的xgb lgb cbt + 我的3-fold nn + online lgb + online ridge

## Winner solutions

### - What is the key to a successful sequence model?

https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/discussion/556859

1: Feeding **one date_id of data as one batch** is the most important part for the successful sequence model in this competition (could be the key to break 0.009)
2: Designing modules or features to combine time-dependency with inter-symbol_ids dependency (real-time market status) smartly could be the key to break 0.01
3: Adding extra features (or smart model design to include this) and using all responders as labels for training could be the key to break 0.011 (just my guess from the shared solutions)
4: No idea how to break 0.012 or even 0.013 (probably smarter online learning strategy instead of date_id by date_id online training)…

### - [Public LB 6th] solution（有代码）

https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/discussion/556542

Link to the code https://github.com/evgeniavolkova/kagglejanestreet
Link to the submission notebook https://www.kaggle.com/code/eivolkova/public-6th-place?scriptVersionId=217330222

一个date_id是一个batch

#### 1. Cross-validation

I used a time-series CV with **two folds. The validation size was set to 200 dates, as in the public dataset. It correlated well with the public LB scores. Additionally, the model from the first fold was tested on the last 200 dates with a 200-day gap to simulate the private dataset scenario.**

#### Model

Time-series GRU with sequence equal to one day. I ended up with two slightly different architectures:

- 3-layer GRU
- 1-layer GRU followed by 2 linear layers with ReLU activation and dropout.

The second model worked better than the first model on CV (+0.001), but the first model still contributed to the ensemble, so I kept it.

MLP, time-series transformers, cross-symbol attention and embeddings didn't work for me.

#### Feature engeneering

I used all original features except for three categorical ones (features 09–11). I also selected 16 features that showed a high correlation with the target and created two groups of additional features:

- Market averages: Averages per `date_id` and `time_id`.
- Rolling statistics: Rolling averages and standard deviations over the last 1000 `time_id`s for each symbol.

Besides that, I added `time_id` as a feature.

I used 4 responders as **auxiliary targets**: `responder_7` and `responder_8`, and two calculated ones:

#### Online Learning

During inference, when new data with targets becomes available, I perform one forward pass to update the model weights with a learning rate of 0.0003. This approach significantly improved the model’s performance on CV (+0.008). Interestingly, for an MLP model, the score without online learning was higher than for the GRU, but lower with online learning.

Updates are performed only with the `responder_6` loss, without auxiliary targets.

### - Tricks to make CatBoost online training great again~

https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/discussion/556544

#### Trick # 1: You don't need to train the model from scratch in one shot, instead you could divide it into multiple trainings.

```python
# version 1: train model in one go
train_data = Pool(train[features_cbt],train[target_col])
cbt_model = cbt.CatBoostRegressor(
    iterations=1500,
    learning_rate=0.03,
    depth=9,
    l2_leaf_reg=0.066,
    bagging_temperature=0.71,
    random_strength=3.6,
    random_seed=42,
    task_type='GPU',
    loss_function='RMSE',
)
cbt_model.fit(train_data, silent=True)

#version 2: divide train into 5 times

cbt_model = []
for i in range(5):
    if i > 0:
        score_increase = cbt_model_i.predict(train[features_cbt])
        train_scores_current = train_scores_current + score_increase
    else:
        train_scores_current = None
train_data = Pool(train[features_cbt],train[target_col], baseline=train_scores_current)
cbt_model_i = cbt.CatBoostRegressor(
    iterations=300,
    learning_rate=0.03,
    depth=9,
    l2_leaf_reg=0.066,
    bagging_temperature=0.71,
    random_strength=3.6,
    random_seed=42,
    task_type='GPU',
    loss_function='RMSE',
)
cbt_model_i.fit(train_data, silent=True)
cbt_model.append(cbt_model_i)
```

#### Trick # 2: You don't need to all the data points from each date_id, but sample a portion.

For each date_id, there are 968 different time_ids from date_id: 677, and for each (date_id, time_id), there is tons of symbol_ids, one assumption here is there might be a lot of redundant info here for GBDT model training. So what I tried is for each date_id, I will sample a portion (say 40% of all records), while using only the most recent 600 date_ids. Interestingly, I almost got the similar model performance, compared to using all the datapoints (still last 600 date_ids wihtout sampling)



**Using those two tricks together with others (anomaly removal and post-process namely uncertainty estimation), I could build a catboost online training pipeline using last 600 days with 180 features while training freq every 7 days to achieve 0.008+ LB score.**

### - Some final remarks

https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/discussion/556686

用了Encoder + decoder + GRU 的architecture。

关键问题，数据non-stationary的解决办法：

**Normalization**
I tried various schemes: signed log transform, global norm, Yeo–Johnson (with fixed/trainable lambda), quantile transform, and EMA with trainable momentum. My takeaway? It doesn’t really matter. The main benefit is numerical stability. Given the non-stationary nature of the signal, I expected Yeo–Johnson or EMA to perform better, as they can adapt quickly to distribution shifts, but…

**Hyperparameters**
Most optimizers in TensorFlow (Adam, AdaMax, etc.) use some form of momentum—they combine and weight gradients from the last N steps to stabilize training (the specifics vary, but the principle is similar). This approach works for stationary signals or use cases with a universal ground truth. We have neither here, and we don’t want the model to keep updating gradients based on outdated patterns. **So, I set the “beta” parameters of Adam to reasonably low values, reducing the effective window size to 2-5 days.**

**这点我看懂了。将Adam中的beta参数设置地很小，因为pytorch默认参数是0.9和0.999,*betas=(0.9, 0.999)*, 也就相当于计算过去10天的梯度移动平均和过去1000天的梯度平方的移动平均，这对于non-stationary太多了。**太细了。。

### - [Public LB 26th] TabM, AutoencoderMLP with online training & GBDT offline models （有代码）

https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/discussion/556610

TabM training code https://www.kaggle.com/code/i2nfinit3y/jane-street-tabm-training?scriptVersionId=217873214
Single TabM online learning : https://www.kaggle.com/code/i2nfinit3y/jane-street-tabm-online-learning

**Inference Code**: You can access our inference code [here](https://www.kaggle.com/code/chronoscop/fork-of-jane-street-rmf-final-submission?scriptVersionId=218125086).
**Training Code**: Our training code is available on GitHub [here](https://github.com/chronoscop/JS-Public-LB-26th-training-code).

Online TabM   + AutoencoderMLP + offline gbdt ridge

Our final solution of 0.0092 lb is ensembling NN models with online learning , GBDT offline models and a ridge model. My part is mainly for TabM model and some GBDT models, which is what I am gonna to talk about. AutoencoderMLP and online learning are [@lechengyan](https://www.kaggle.com/lechengyan) 's part and [@chronoscop](https://www.kaggle.com/chronoscop) is in charge of one of XGB.

#### 1. Cross-Validation

I simply used the last 120 dates as my validation and it shows good correlations with LB.

### - [Public LB 12th] Competition Wrap-up: （Private掉到29 2025-3-26）

https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/discussion/556548

- For models, I designed two different architectures using basic ingredients including GRU, MLP and **Transformer (symbol-wise attention).** Under each architecture, feature maps were ensemble in two ways, resulting in 4 different models.
- Features are the 79 raw features excluding `9, 10, 11`, `time_id`, `weights`, as well as mean and std of the lagged responders. Missing values were filled with zeros. All responders were used as targets (instead of only Responder_6). As [@eivolkova](https://www.kaggle.com/eivolkova) pointed out, using auxiliary targets can greatly boost both CV & LB.
- Models were validated using the last 120 days, with both offline and online mode. Training sets includes three settings, i.e. 978 days, 800 days and 600 days. Eventually only models trained with the 978 and 800 days were used in the final ensemble (8 models).
- **Online learning** was designed to update the model on a daily basis, using a similar setting as the training. Unlike [@eivolkova](https://www.kaggle.com/eivolkova) 's solution, I did not differentiate the responder_6 loss and auxiliary targets loss during the online update. The model updating is quite fast. It was about 0.5~0.7 sec per model per day. A full online training using every 120 or 200 days could further boost the score, however I did not implement it as it will complicate the whole pipeline quite a lot. The major concern is the 1-min limit.

### - Symbol Cross-Attention Animations（有一点代码）

I used batches of **(Symbol, Time, Feature)**, one batch per day.

对于Attention层来说，输入shape是# Shape: (B, L, FT)，输出shape还是# Shape: (B, L, FT)，区别在于新的# Shape: (B, L, FT)是经过attention了其他symbol之后得到的。

```python
class SymbolCrossAttention(nn.Module):
    def __init__(self, feature_dim, projection_dim):
        super(SymbolCrossAttention, self).__init__()
        self.feature_dim = feature_dim
        self.projection_dim = projection_dim
        # Learnable projections for computing similarity
        mlp_hidden_dim = 256
        # Define MLP for queries
        layers = [nn.Linear(feature_dim, mlp_hidden_dim), nn.ReLU()]
        for _ in range(0):  # Add hidden layers
            layers.extend([nn.Linear(mlp_hidden_dim, mlp_hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(mlp_hidden_dim, projection_dim, bias=False))
        self.query_proj = nn.Sequential(*layers)

        # Define MLP for keys (can share weights with queries if desired)
        layers = [nn.Linear(feature_dim, mlp_hidden_dim), nn.ReLU()]
        for _ in range(0):  # Add hidden layers
            layers.extend([nn.Linear(mlp_hidden_dim, mlp_hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(mlp_hidden_dim, projection_dim, bias=False))
        self.key_proj = nn.Sequential(*layers)

    def forward(self, x):
        # Input shape: (B, L, FT)
        B, L, FT = x.shape
        assert FT == self.feature_dim, "Feature dimension mismatch"
        # Project features into the high-dimensional space
        queries = self.query_proj(x)  # Shape: (B, L, P)
        keys = self.key_proj(x)      # Shape: (B, L, P)
        # Compute similarity scores for all timesteps in parallel
        # Step 1: Reshape for batch-level interaction
        queries = queries.permute(1, 0, 2)  # Shape: (L, B, P)
        keys = keys.permute(1, 0, 2)        # Shape: (L, B, P)
        # Step 2: Compute similarity matrix: (L, B, B)
        similarity = torch.bmm(queries, keys.transpose(1, 2))  # Dot product: (L, B, B)
        # Step 3: Scale the similarity scores by 1 / sqrt(P)
        similarity = similarity / (self.projection_dim ** 0.5)  # Scale by sqrt(P)
        # Step 3: Normalize with softmax over rows
        attention_weights = F.softmax(similarity, dim=-1)  # Shape: (L, B, B)
        # Step 4: Compute weighted sum of features
        # Reshape original features for batch-level interaction
        x = x.permute(1, 0, 2)  # Shape: (L, B, FT)
        result = torch.bmm(attention_weights, x)  # Weighted sum: (L, B, FT)
        # Step 5: Reshape back to original shape
        result = result.permute(1, 0, 2)  # Shape: (B, L, FT)
        return result


# Main model: __init__
        self.batch_attention = SymbolCrossAttention(feature_dim, projection_dim)

# Main model: in forward()
        attention_result = self.batch_attention(features)
        features = torch.cat([
            features,
            attention_result,
            features - attention_result,
        ], dim=-1)
```

### - Lightgbm online training ideas

和catboost那个一样，关键在于分阶段训练，每阶段的Init_score用保存好的上一阶段的score，这样就不需要从头开始预测score了。

A couple of issues with lightgbm in the online learning settings that I managed to solve were:

- the lack of timeout parameter in the training function
- the linear increase in startup time when continuing a partial training.

#### Timeout

The timeout issue is solvable with a simple callback

``

```
class LGBMTimeoutCallback:
    def __init__(self, timeout=None):
        self.timeout = timeout
        self.t0 = time.time_ns()
    def __call__(self, env):
        dt = (time.time_ns() - self.t0) / 1e9
        if self.timeout is not None and dt >= self.timeout:
            raise lightgbm.EarlyStopException(env.iteration,  env.evaluation_result_list)
```

#### Partial training

The continuation of partial training is a bit trickier. Let's compare with a simple implementation

~~~python
```
dataset = lightgbm.Dataset(data=x, label=y, free_raw_data=False)
model = None
for _ in range(10):
    model = lightgbm.train(
        params=params,
        train_set=dataset,
        num_boost_round=10,
        init_model=model,
        keep_training_booster=True
    )
```
~~~



A possible solution is to provide a custom `Dataset` implementation and manually update the `init_score` adding only the score of the latest trees, **instead of recomputing for all the trees every time**

~~~python
```
class LGBMDataset(lightgbm.Dataset):
    def _set_init_score_by_predictor(self, predictor, data, used_indices):
        if self.init_score is None:
            return super()._set_init_score_by_predictor(predictor, data, used_indices)
        return self

dataset = LGBMDataset(data=x, label=y, free_raw_data=False)
model = None
for _ in range(10):
    num_trees_before = 0 if model is None else model.num_trees()
    model = lightgbm.train(
        params=params,
        train_set=dataset,
        num_boost_round=10,
        init_model=model,
        keep_training_booster=True
    )
    init_score = model.predict(dataset.data, start_iteration=num_trees_before)
    if dataset.init_score is not None:
        init_score = dataset.init_score + init_score.reshape(dataset.init_score.shape)
    dataset.set_init_score(init_score)
```
~~~

### - [Public LB 13th] Our Journey to 0.0096

https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/discussion/556701

post的字太多了。。先不看



### - [Public LB 17th] Solution

https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/discussion/556541

#### Model Architecture

Architecture consisted of a 50/50 ensemble between 3 layers transformer encoder with attention over all symbol_id and 3 layers transformer encoder with attention over all time_id and learnable positional encoding. I also used GELU activations to prevent dead neurons caused by ReLU so during online learning parameters could come back into play if necessary. Also tried SiLU and PReLU but GELU was best.

#### Features

`features = [f'feature_{i:02d}' for i in range(0, 79)] + ['time_id', 'weight']`. Also added signed log features for train stability. I originally tried hand picking which features I should use the log version for and which should have the original version but eventually gave up and passed in all original and all log features to the model.

Feature里加了Log feature。

#### Online Learning (main score boost)

At first I tried a simple idea of doing a single update step on the latest day of data given by the API. This barely helped. I then ran an experiment to see how much I could improve my score on the last 20-30 days of validation data if I fine tuned my model on the few weeks (can't exactly remember how many) just prior to that. After playing around with some settings I found that I could train on this data for around 7 epochs and it would give me a significant boost in the CV score for the last days. This gave me a starting point. I decided to train on the last 7 days every day in order to update the model. This way the model is trained 7 times on each new day (7 "epochs"). I used lr=1e-4 instead of the original 1e-3 and everything else about the setup is exactly the same as my original train setup.

刚开始用最新一天的数据训练，效果不行。后改为用前7天的数据训练 （应该是一个epoch），学习率由原来的1e-3变为1e-4。