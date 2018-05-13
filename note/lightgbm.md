# LightGBM
일반적인 GBDT (Gradient-Boosting-Decision-Tree)를 GOSS (Gradient-based One-Side-Sampling)와 EFB (Exclusive Feature Bundling)를 통해 발전시킨 모델

[LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf) - 관련 논문
[참고 및 이미지 출처](https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc)

## 목차
1. LightGBM을 쓰는 이유
1. Gradient-Boosting-Decision-Tree
1. Gradient-based One-Side-Sampling
1. EFB
1. parameters

## LightGBM을 쓰는 이유
1. 굉장히 빠릅니다.
1. 다른 라이브러리에 비해 메모리 효율적입니다. (더 큰 데이터 분석 가능)
1. [XGboost와의 성능 비교](https://github.com/Microsoft/LightGBM/blob/master/docs/Experiments.rst#comparison-experiment)

## Gradient-Boosting-Decision-Tree
GBDT는 이름과 같이 의사결정 나무를 weak classifier로 사용하는 부스팅 기법의 앙상블 모델입니다. Lightgbm의 goss모델은 이 모델을 발전시킨 형태입니다.
- Boosting이란?
    부스팅 방법은 미리 정해진 모형 집합을 사용하는 것이 아니라 단계적으로 모형 집합에 포함할 개별 모형을 선택합니다. 부스팅 방법에서는 성능이 떨어지는 개별 모형을 사용하며, 이를 약 분류기(weak classifier) 라고 합니다.

- Gradient Boost란?
    그레디언트 부스트 모형은 최적화에 사용되는 gradient descent 방법을 응용한 모형입니다. 함수 $f(x)$를 최소화하는  $x$는 다음과 같이 gradient descent 방법으로 찾을 수 있습니다.
    $$ x_m = x_{m - 1} - \alpha_m \dfrac{df}{dx} $$

    그레디언트 부스트 모형에서는 오차 함수 또는 손실 함수(loss function)  $L(y,C_{m-1})$을 최소화하는 weak classifier  $k_m $은  $\frac{-dL(y,C_{m-1})}{dC_{m-1}}$ 임을 알 수 있습니다.

    $$C_{m} = C_{m-1} - \alpha_m \dfrac{dL(y, C_{m-1})}{dC_{m-1}} = C_{m-1} + \alpha_m k_m$$

    따라서 그레디언트 부스트 모형에서는 다음과 같은 과정을 반복합니다.
    1. $-\tfrac{dL(y, C_m)}{dC_m}$을 타겟으로 하는 week classifier $k_m$을 찾는다.
    1. $\left( y - (C_{m-1} + \alpha_m k_m) \right)^2$를 최소화하는 step size $\alpha_m$을 찾는다.
    1. $C_m = C_{m-1} + \alpha_m k_m$ 최종 모형을 갱신한다.


## Gradient-based One-Side-Sampling
Light GBM의 GOSS는 의사 결정 나무의 모든 잎(node)들을 차례대로 보는 것이 아니라, 정보량이 높은(엔트로피가 높은, 분류가 덜 된) 잎을 먼저 본다고 할 수 있습니다. *실제로는, 샘플링을 할 때 정보량이 적은 잎들을 제외함으로써 계산 속도를 높히는 방식입니다.* 따라서 같은 갯수의 잎을 가진다고 했을 때, 일반적인 level-wise 방식보다 loss를 줄일 수 있다고 합니다.
- 장점:
1. GBDT에 비해 빠릅니다.
2. GBDT에 비해 훨씬 적은 데이터 사이즈로 비슷한 성능을 낼 수 있습니다.

- 단점:
1. 과 최적화되기 쉽습니다.

<p align="center">
  <img src="https://github.com/Moons08/LightGBM-tutorial-Fraud_Detection/blob/master/img/leaf-wise.png?raw=true" width="700">
</p>
<p align="center">
  <img src="https://github.com/Moons08/LightGBM-tutorial-Fraud_Detection/blob/master/img/level-wise.png?raw=true" width="700">
</p>

## EFB

High-dimensional data are usually very sparse. The sparsity of the feature space provides us a possibility of designing a nearly lossless approach to reduce the number of features. Specifically, in a sparse feature space, many features are mutually exclusive, i.e., they never take nonzero values simultaneously. We can safely bundle exclusive features into a single feature (which we call an exclusive feature bundle). By a carefully designed feature scanning algorithm, we can build the same feature histograms from the feature bundles as those from individual features. In this way, the complexity of histogram building changes from O(#data × #feature) to O(#data × #bundle), while #bundle << #feature. Then we can significantly speed up the training of GBDT without hurting the accuracy. In the following, we will show how to achieve this in detail. There are two issues to be addressed. The first one is to determine which features should be bundled together. The second is how to construct the bundle.


## Parameters

**control**
- max_depth
    의사결정 나무의 깊이(depth, level)를 제한합니다. 과최적화를 막기 위해 이용됩니다. 과최적화가 조금이라도 된 느낌이라면 max_depth를 먼저 낮게 해보는 것이 좋습니다.

- min_data_in_leaf
    각 잎(노드)에 포함되어있는 데이터의 최소 갯수 입니다. 이 값 이하로는 잎이 나누어지지 않습니다. 디폴트 값은 20으로 설정되어있습니다. 이 또한 과최적화 방지를 위해 이용되는 파라미터입니다.

- early_stopping_round
    설정해 놓은 값의 iteration 동안 성능이 좋아지지 않으면 training을 종료합니다. 최적점에 도달한 이후 반복되는 불필요한 training을 방지할 수 있습니다. 이 값을 너무 낮게 잡으면 local optima에 빠진 채로 학습이 종료될 수도 있습니다.

- feature_fraction
    부스팅 방식을 랜덤 포레스트로 사용할 경우 이용할 수 있습니다. 예를 들어, 0.8은 각 iteration에서 의사결정 나무를 만들 때 80%의 파라미터(독립 변수)를 랜덤하게 설정한다는 의미입니다.  

- bagging_fraction
    specifies the fraction of data to be used for each iteration and is generally used to speed up the training and avoid overfitting.

- lambda: lambda specifies regularization. Typical value ranges from 0 to 1.

- min_gain_to_split: This parameter will describe the minimum gain to make a split. It can used to control number of useful splits in tree.

- max_cat_group: When the number of category is large, finding the split point on it is easily over-fitting. So LightGBM merges them into ‘max_cat_group’ groups, and finds the split points on the group boundaries, default:64

**core**

- application
    1. regression
    1. binary
    1. multiclass

- Boosting
    1. gbdt : traditional Gradient Boosting Decision Tree
    1. rf : random forest
    1. dart: Dropouts meet Multiple Additive Regression Trees
    1. goss: Gradient-based One-Side Sampling
