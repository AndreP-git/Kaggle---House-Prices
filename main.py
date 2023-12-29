import pandas as pd
import tensorflow as tf 
from d2l import tensorflow as d2l


class KaggleHouse(d2l.DataModule):
    def __init__(self, batch_size, train=None, val=None):
        super().__init__()
        self.save_hyperparameters()
        if self.train is None:
            self.raw_train = pd.read_csv("./data/train.csv")
            self.raw_val = pd.read_csv("./data/test.csv")
            
    def preprocess(self):
        # Remove the ID and label columns
        label = 'SalePrice'
        features = pd.concat(
            (self.raw_train.drop(columns=['Id', label]),
            self.raw_val.drop(columns=['Id'])))
        # Standardize numerical columns
        numeric_features = features.dtypes[features.dtypes!='object'].index
        features[numeric_features] = features[numeric_features].apply(
            lambda x: (x - x.mean()) / (x.std()))
        # Replace NAN numerical features by 0
        features[numeric_features] = features[numeric_features].fillna(0)
        # Replace discrete features by one-hot encoding
        features = pd.get_dummies(features, dummy_na=True)
        # Save preprocessed features
        self.train = features[:self.raw_train.shape[0]].copy()
        self.train[label] = self.raw_train[label]
        self.val = features[self.raw_train.shape[0]:].copy()
        

    def get_dataloader(self, train):
        label = 'SalePrice'
        data = self.train if train else self.val
        if label not in data: return
        get_tensor = lambda x: tf.constant(x.values.astype(float),
                                        dtype=tf.float32)
        # Logarithm of prices
        tensors = (get_tensor(data.drop(columns=[label])),  # X
        tf.reshape(tf.math.log(get_tensor(data[label])), (-1, 1)))  # Y
        return self.get_tensorloader(tensors, train)
    
def k_fold_data(data, k):
    rets = []
    fold_size = data.train.shape[0] // k
    for j in range(k):
        idx = range(j * fold_size, (j+1) * fold_size)
        rets.append(KaggleHouse(data.batch_size, data.train.drop(index=idx),
                                data.train.loc[idx]))
    return rets

def k_fold(trainer, data, k, lr):
    val_loss, models = [], []
    for i, data_fold in enumerate(k_fold_data(data, k)):
        model = d2l.LinearRegression(lr)
        model.board.yscale='log'
        if i != 0: model.board.display = False
        trainer.fit(model, data_fold)
        val_loss.append(float(model.board.data['val_loss'][-1].y))
        models.append(model)
    print(f'average validation log mse = {sum(val_loss)/len(val_loss)}')
    return models

if __name__ == "__main__":
    
    # Load data
    data = KaggleHouse(batch_size=64)
    print(data.raw_train.shape)
    print(data.raw_val.shape)
    print(data.raw_train.iloc[:4, [0, 1, 2, 3, -3, -2, -1]])
    
    # Preprocess data
    data.preprocess()
    print(f"train shape (after preprocessing): {data.train.shape}")
    
    # Model selection
    trainer = d2l.Trainer(max_epochs=10)
    models = k_fold(trainer, data, k=5, lr=0.01)
    
    # Making predictions
    preds = [model(tf.constant(data.val.values.astype(float), dtype=tf.float32))
         for model in models]
    # Taking exponentiation of predictions in the logarithm scale
    ensemble_preds = tf.reduce_mean(tf.exp(tf.concat(preds, 1)), 1)
    submission = pd.DataFrame({'Id':data.raw_val.Id,
                            'SalePrice':ensemble_preds.numpy()})
    submission.to_csv('submission.csv', index=False)
    
    # END MAIN
    