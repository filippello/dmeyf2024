import pandas as pd
data = pd.read_csv('/home/fililoco/buckets/b1/datasets/competencia_02_baja.csv')
mes_train = ['202008','201908','22108']
newdata = data[data['foto_mes'].isin(mes_train)]
data.to_csv('/home/fililoco/buckets/b1/datasets/competencia_02_08.csv',index=False)