import pandas as pd
data = pd.read_csv('/Users/federicofilippello/Downloads/datasets_competencia_02_baja.csv')
mes_train = [202008,201908,202108]
newdata = data[data['foto_mes'].isin(mes_train)]
newdata.to_csv('/Users/federicofilippello/Downloads/competencia_02_08.csv',index=False)