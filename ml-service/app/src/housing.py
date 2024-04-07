from enum import Enum

from catboost import CatBoostRegressor
import pandas as pd
from pydantic import BaseModel, Field


MODEL_PATH = './ml-models/'


class HousingType(str, Enum):
    single_room = 'Однокомнатная'
    double_room = 'Двухкомнатная'
    triple_room = 'Трехкомнатная'
    quadro_room = 'Четырехкомнатная'
    multi_room = 'Многокомнатная'


class CityArea(str, Enum):
    ordzh = 'Орджоникидзевский'
    lenin = 'Ленинский'
    right = 'Правобережный'
    left = 'Орджоникидзевский (левый берег)'
    none = 'None'


class Street(str, Enum):
    street_1 = 'Проспект Ленина'
    street_2 = 'Карла Маркса'
    street_3 = 'Советская'
    street_4 = 'Зеленый Лог'
    street_5 = 'Труда'
    street_6 = '50-летия Магнитки'
    street_7 = 'Газеты Правда'
    street_8 = 'Курортная'
    street_9 = 'Грязнова'
    street_10 = 'Суворова'
    street_11 = 'Московская'
    street_12 = 'Уральская'
    street_13 = 'Коробова'
    street_14 = 'Металлургов'
    street_15 = 'Октябрьская'
    street_16 = 'Тевосяна'
    street_17 = 'Ворошилова'
    none = 'None'


class Layout(str, Enum):
    layout_1 = 'улучшенная'
    layout_2 = 'брежневка'
    layout_3 = 'нестандартная'
    layout_4 = 'старой планировки'
    layout_5 = 'раздельная'
    layout_6 = 'хрущевка'
    none = 'None'


class Housing(BaseModel):
    type: HousingType
    area: CityArea
    street: Street
    layout: Layout
    floor: int = Field(gt=0, default=1)
    floors: int = Field(gt=0, default=1)
    whole_area: float = Field(gt=0, default=50.)
    living_area: float = Field(gt=0, default=40.)
    kitchen_area: float = Field(gt=0, default=10.)


params_map = {'Тип квартиры': 'type',
              'Район': 'area',
              'Улица': 'street',
              'Этаж': 'floor',
              'Планировка': 'layout',
              'о': 'whole_area',
              'ж': 'living_area',
              'к': 'kitchen_area',
              'Этажей': 'floors'}


def predict(housings: list[Housing]):
    # Convert pydantic model to python list of dicts
    for i, housing in enumerate(housings):
        housing = housing.model_dump()
        for k, v in params_map.items():
            if isinstance(housing[v], Enum):
                housing[k] = housing[v].value
            else:
                housing[k] = housing[v]
            del housing[v]
        housings[i] = housing

    # Convert list of dicts to pandas DataFrame
    x_test = pd.DataFrame(housings)

    # Load model
    model_file = MODEL_PATH + 'housing_catboost.cbm'
    model = CatBoostRegressor()
    model.load_model(model_file)

    # Predict
    return model.predict(x_test).tolist()
