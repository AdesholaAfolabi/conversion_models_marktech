import pickle,os,glob,yaml
from FeatureEngineering import feature_engineering as fe
import pandas as pd
import numpy as np
import s3fs
import pyarrow.parquet as pq

fs = s3fs.S3FileSystem()
config = yaml.safe_load(open("config.yml"))
path = "s3://data-lake-v2/processed_batch_data/clean/flat_demography/year=2020/month=03/"
numerical_features = config['numerical_features']
input_column = config['input_column']
reindex_lists = config['reindex_column']

class ScoringService(object):
    model = None                # Where we keep the model when it's loaded
    preprocessor = None         # Where we keep the preprocessor when it's loaded
    
    @classmethod
    def get_model(cls,model_path):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            # load the model from disk
            with open(model_path, 'rb') as file:
                cls.model = pickle.load(file)
        return cls.model

    @classmethod
    def predict(cls, input, model_path):    
        clf = cls.get_model(model_path)
        predictions = clf.predict_proba(input)[:,1]
        return predictions
    
    @classmethod
    def get_preprocessor(cls,preprocessor_path):
        """Get the preprocessor object for this instance, loading it if it's not already loaded."""
        if cls.preprocessor == None:
            with open(preprocessor_path, 'rb') as file:
                cls.preprocessor = pickle.load(file)
        return cls.preprocessor
    
    @classmethod
    def score_db(cls,data_path,input_column,preprocessor_path,model_path):
        counter = 0
        #for file in glob.glob(data_path+'/'+'part*'):
        for file in fs.ls(data_path):
            print (file)
            #data = pd.read_parquet("s3://"+file, engine='pyarrow')
            data = pq.ParquetDataset('s3://'+file, filesystem=fs).read_pandas(columns = input_column).to_pandas()
            msisdn = data['msisdn']
            operator = data.pop('operator')
            data['loan_propensity'] = pd.to_numeric(data['loan_propensity'])
            f = fe(data,numerical_features)
            f._normalize_data(target_enc=False)
            f._preprocessing()
            print(data.columns)
            for col in data.columns:
                if col not in numerical_features:
                    data[col] = data[col].apply(str)
            preprocessor = cls.get_preprocessor(preprocessor_path)
            data = preprocessor.transform(data)
            print('invoking model endpoint')
            # Do the prediction
            predictions = np.round(cls.predict(data,model_path),3)
            row_count = len(predictions)
            print(f'Returned predictions for {row_count} rows')

            out = 'out'+ str(counter) + '.csv'
            bytes_to_write = pd.DataFrame({'msisdn':msisdn,'score':predictions}).to_csv(None,index=False,
                                                                             sep=',').encode()

            with fs.open("s3://datateam-ml/predictions/fsi_ctr_rf/{}".format(out), 'wb') as f:
                f.write(bytes_to_write)

            counter +=1
            print('Done. Check the predictions path')
            
            
directory = os.getcwd()
model_path = os.path.join(directory,'saved_model/RF.sav')
preprocessor_path = os.path.join(directory,'saved_model/pipeline.pkl')

if __name__ == '__main__':
    ScoringService.score_db(path,input_column,preprocessor_path,model_path)