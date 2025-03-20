# NYC Apartment Price Prediction with XGBoost

Este projeto utiliza o modelo XGBoost para prever preços de apartamentos em Nova York. Inclui análise exploratória e um modelo treinado salvo.

## Estrutura do Projeto

- `Análise Exploratória.ipynb`: Notebook com a análise exploratória dos dados.
- `requirements.txt`: Lista de dependências necessárias para executar o projeto.
- `teste_indicium_precificacao.csv`: Dataset de teste utilizado para prever.
- `xgb_model.pkl`: Modelo treinado salvo com XGBoost.

## Instalação

1. Clone o repositório:
   ```
   git clone https://github.com/LiliaDieZeitgeist/nyc-apt-price-prediction-with-xgboost.git
   cd nyc-apt-price-prediction-with-xgboost
   ```

2. Crie um ambiente virtual e ative-o:
   ```
   python -m venv venv
   source venv/bin/activate  # No Windows, use: venv\Scripts\activate
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

## Execução

1. Abra o notebook Jupyter para explorar os dados e executar o modelo:
   ```
   jupyter notebook
   ```

2. Execute o arquivo `Análise Exploratória.ipynb`.

3. Para prever usando o modelo salvo:
   - Certifique-se de que os dados de teste estão formatados corretamente.
   - Converta os dados de teste para `DMatrix` e use o modelo para previsões:
     ```
     import xgboost as xgb
     import pandas as pd

     teste = pd.read_csv('teste_indicium_precificacao.csv')
     dtest = xgb.DMatrix(teste)

     bst = xgb.Booster()
     bst.load_model('xgb_model.pkl')
     y_pred = bst.predict(dtest)
     print(y_pred)
     ```
