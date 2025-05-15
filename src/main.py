import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sqlalchemy import create_engine, Table, Column, Float, MetaData
import os

def data_process():
    # 1. Procesamiento de datos
    df = pd.read_csv('./data/sales_data.csv')
    df = df.dropna()
    return df

def linear_regression(df,x_list,y_col):
    # 2. Modelado de regresión
    X = df[x_list]
    y = df[y_col]
    
    model = LinearRegression()
    model.fit(X, y)
    
    y_pred = model.predict(X)
    r_squared = r2_score(y, y_pred)
    print(f"R²: {r_squared:.4f}")
    price_coef = model.coef_[0]
    return model, y_pred, r_squared, price_coef

def calculate_elasticity(df, price_coef, col_name_price, col_name_demand):
    # 3. Cálculo de elasticidad
    # Para calcular la elasticidad precio de la demanda con una regresión lineal, 
    # se necesita la pendiente de la regresión lineal y 
    # los valores promedio de la cantidad demandada y el precio. 
    # La fórmula para calcular la elasticidad precio de la demanda es la 
    # pendiente de la regresión lineal multiplicada por el precio promedio dividido
    #  por la cantidad promedio.

    mean_price = df[col_name_price].mean()
    mean_quantity = df[col_name_demand].mean()
    elasticity = price_coef * (mean_price / mean_quantity)
    
    # Determinar si es elástica o inelástica
    elasticity_type = 'elástica' if abs(elasticity) > 1 else 'inelástica'
    
    print(f"Coeficiente de precio: {price_coef:.4f}")
    print(f"Elasticidad precio: {elasticity:.4f} ({elasticity_type})")

    return elasticity

def postgresql_storage(r_squared, price_coef, elasticity):
    # 4. Almacenamiento en PostgreSQL
    db_config = {
        'host': os.environ.get('DB_HOST', 'db'),  # Valor por defecto 'db' para Docker
        'database': os.environ.get('DB_NAME', 'soho_db'),
        'user': os.environ.get('DB_USER', 'user'),
        'password': os.environ.get('DB_PASS', 'password')
    }
    print("\nConfiguración DB desde Docker:")
    print(f"Host: {db_config['host']}")
    print(f"DB: {db_config['database']}")
    print(f"User: {db_config['user']}")
    db_url = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}"
    print(db_url)
    engine = create_engine(db_url)
    # Crear tabla si no existe
    metadata = MetaData()
    results_table = Table('results', metadata,
        Column('r_squared', Float),
        Column('price_coefficient', Float),
        Column('price_elasticity', Float)
    )
    metadata.create_all(engine)
    
    # Insertar resultados
    with engine.connect() as conn:
        conn.execute(results_table.insert().values(
            r_squared=r_squared,
            price_coefficient=price_coef,
            price_elasticity=elasticity
        ))
        # conn.commit()   

def main():
    df = data_process()
    model, y_pred, r_squared, price_coef = linear_regression(df,
                                                 ['price', 'competitor_price'],
                                                 'quantity_sold')
   
    elasticity = calculate_elasticity(df, price_coef, 'price', 'quantity_sold')
    postgresql_storage(r_squared, price_coef, elasticity)
if __name__ == "__main__":
    main()