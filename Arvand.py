import numpy as np
import joblib
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# Заголовок
st.subheader('Добро пожаловать в скоринговую систему банка Arvand')

# Загрузка моделей и данных





with open("lr_model.pkl", "rb") as pickle_in:
    regression1 = joblib.load(pickle_in)

with open("model_gb.pkl", "rb") as pickle_in:
    regression2 = joblib.load(pickle_in)

# Создание DataFrame с данными
data1 = pd.DataFrame({'Married': ['Беоила', 'Оиладор', 'Бевамард (бевазан)', 'Чудошуда']})
data2 = pd.DataFrame({'Nationality': ['Узбек', 'Точик', 'Тотор', 'Рус', 'Киргиз', 'Украин', 'Другие', 'Карис', 'Карачои']})
data3 = pd.DataFrame({'Educ': ['Аспирантура', 'Миёна', 'Миёнаи махсус', 'Миёнаи нопурра', 'Оли', 'Олии нопурра']})

    
def issue_a_loan(FamilySize, Loan_Amount, Loan_Term,
                 Monthly_repayment_amount_according_to_schedule, Grace_period,
                 Capital, Asset, Days_overdue, Number_of_overdue,
                 Lending_stage, Gross_profit,
                 Net_Profit, Age, Married_encoded, isFemale, nationality_encoded, educ, Filial_code,
                 Region_code, Direction_of_activity, Currency_code, Pledge_code,
                 business_experience):
    # Преобразование business_experience в числовой формат
    business_experience = int(business_experience)

    # Преобразование всех переменных в числовой формат
    input_data = [
        FamilySize, Loan_Amount, Loan_Term,
        Monthly_repayment_amount_according_to_schedule, Grace_period,
        Capital, Asset, Days_overdue, Number_of_overdue,
        Lending_stage, Gross_profit,
        Net_Profit, Age, Married_encoded, isFemale, nationality_encoded, educ, Filial_code,
        Region_code, Direction_of_activity, Currency_code, Pledge_code,
        business_experience
    ]

    # Объединение закодированных данных
    input_data.extend(Married_encoded.values.flatten())
    input_data.extend(Nationality_encoded.values.flatten())
    input_data.extend(Educ_encoded.values.flatten())
    
                     
    # Преобразуем в массив numpy и делаем предсказание
    input_array = np.array(input_data).reshape(1, -1)
    prediction1 = regression1.predict(input_array)
    prediction2 = regression2.predict(input_array)
    total_pred = (prediction1 + prediction2) / 3
    
    prediction2_1 = regression1.predict_proba(input_array)
    prediction2_2 = regression2.predict_proba(input_array)
    total_pred2 = (prediction2_1 + prediction2_2) / 3
    return total_pred, total_pred2


def Delays_days(FamilySize, Loan_Amount, Loan_Term,
                 Monthly_repayment_amount_according_to_schedule, Grace_period,
                 Capital, Asset, Days_overdue, Number_of_overdue,
                 Lending_stage, Gross_profit,
                 Net_Profit, Age, Married_encoded, isFemale, nationality_encoded, educ, Filial_code,
                 Region_code, Direction_of_activity, Currency_code, Pledge_code,
                 business_experience):
    # Исправлено: добавлено преобразование business_experience в числовой формат
    business_experience = int(business_experience)

    input_data = [
        FamilySize, Loan_Amount, Loan_Term,
        Monthly_repayment_amount_according_to_schedule, Grace_period,
        Capital, Asset, Days_overdue, Number_of_overdue,
        Lending_stage, Gross_profit,
        Net_Profit, Age, Married_encoded, isFemale, nationality_encoded, educ, Filial_code,
        Region_code, Direction_of_activity, Currency_code, Pledge_code,
        business_experience
    ]

    # Объединение закодированных данных
    input_data = pd.concat([pd.Series(input_data), Married_encoded, Nationality_encoded, Educ_encoded], axis=0)

    # Преобразуем в массив numpy и делаем предсказание
    input_array = np.array(input_data).reshape(1, -1)
    reg1 = regression1.predict(input_array)
    return reg1  

def Credit_sum(FamilySize, Loan_Amount, Loan_Term,
                 Monthly_repayment_amount_according_to_schedule, Grace_period,
                 Capital, Asset, Days_overdue, Number_of_overdue,
                 Lending_stage, Gross_profit,
                 Net_Profit, Age, Married_encoded, isFemale, nationality_encoded, educ, Filial_code,
                 Region_code, Direction_of_activity, Currency_code, Pledge_code,
                 business_experience):
    # Исправлено: добавлено преобразование business_experience в числовой формат
    business_experience = int(business_experience)

    input_data = [
        FamilySize, Loan_Amount, Loan_Term,
        Monthly_repayment_amount_according_to_schedule, Grace_period,
        Capital, Asset, Days_overdue, Number_of_overdue,
        Lending_stage, Gross_profit,
        Net_Profit, Age, Married_encoded, isFemale, nationality_encoded, educ, Filial_code,
        Region_code, Direction_of_activity, Currency_code, Pledge_code,
        business_experience
    ]

    # Объединение закодированных данных
    input_data = pd.concat([pd.Series(input_data), Married_encoded, Nationality_encoded, Educ_encoded], axis=0)

    # Преобразуем в массив numpy и делаем предсказание
    input_array = np.array(input_data).reshape(1, -1)
    reg2 = regression2.predict(input_array)    
    return reg2
                   
# Основная функция
def main():
    FamilySize = st.number_input('Сколько у вас членов семьи?', step=1, value=0)
    Loan_amount = st.number_input('На какую сумму хотите взять кредит?', step=1, value=0)
    Loan_term = st.number_input('На какой срок вы хотите взять кредит(месяц)?', step=1, value=0) 
    Monthly_repayment_amount_according_to_schedule = st.number_input('Ежемесячная сумма погашения?', step=1, value=0) 
    Grace_period = st.number_input('Льготный период?', step=1, value=0)
    Capital = st.number_input('Ваш начальный капитал?', step=1, value=0)
    Asset = st.number_input('Ваш актив?', step=1, value=0)
    Days_overdue = st.number_input('Дни просрочки?', step=1, value=0)
    Number_of_overdue = st.number_input('Дни количество дней просрочки?', step=1, value=0)
    Lending_stage = st.number_input('Какой раз вы уже получаете кредит?', step=1, value=0)
    Gross_profit = st.number_input('Валовая прибыль', step=1, value=0)
    Net_Profit = st.number_input('Чистая прибыль', step=1, value=0)
    Age = st.number_input('Сколько вам лет? запишите ваш год рождения в формате ', step=1, value=0)


    # Создаем dummy-столбцы со значениями False для всех категорий
    Married_encoded = pd.get_dummies(data1, prefix='', prefix_sep='').astype(bool)
    # Получаем выбор пользователя
    selected_Married = st.selectbox('Введите ваше семейное положение:', data1)
    # Если выбор сделан, обновляем соответствующий dummy-столбец в True
    if selected_Married is not None:
        selected_Married = Married_encoded[selected_Married].astype(bool)
    
    isFemale = st.radio("Укажите свой пол:", ['Мужской', 'Женский'])
    if sex == 'Мужской':
        Мужской = 0
    else:
        Женский = 1

# Создаем dummy-столбцы со значениями False для всех категорий
    nationality_encoded = pd.get_dummies(data2, prefix='', prefix_sep='').astype(bool)
    # Получаем выбор пользователя
    selected_nationality = st.selectbox('Введите национальность:', data2)
    # Если выбор сделан, обновляем соответствующий dummy-столбец в True
    if selected_nationality is not None:
        nationality_encoded = nationality_encoded[selected_nationality].astype(bool)


    educ =  ['Аспирантура', 'Оли', 'Миёнаи махсус', 'Олии нопурра', 'Миёна', 'Миёнаи нопурра']
    education = st.selectbox('Уровень образования:', educ)
    label_encoder = LabelEncoder()
    encoded_educ = label_encoder.fit_transform(educ)
    Education = encoded_educ[educ.index(education)]



    

    Filial_code = st.radio("Укажите филиал банка, в котром вы получаете кредит:", ['Истаравшан', 'Хучанд', 'Ч. Расулов', 'Душанбе', 'Исфара', 'Панчакент'])
    if filial == 'Истаравшан':
        Истаравшан = 0
    elif filial == 'Хучанд':
        Хучанд = 1
    elif filial == 'Ч. Расулов':
        Ч. Расулов = 2
    elif filial == 'Душанбе':
        Душанбе = 3
    elif filial == 'Исфара':
        Исфара = 4
    else:
        Панчакент = 5
        
    Region_code = st.radio("Укажите регион, в котром вы получаете кредит:", ['Шахристон', 'Гули сурх', 'Худжанд-Центр', 'Спитамен', 'Шарк', 'Мархамат', 'Душанбе', 'Навкент',
                           'Кистакуз', 'Худжанд-Панчшанбе', 'Бустон', 'Истаравшан-филиал', 'Рудаки', 'Ашт', 'Калининобод',
                           'Сино', 'Исфара', 'Хисор', 'Зафаробод', 'Ничони', 'Вахдат', 'Мехнатобод', 'Уяс', 'Дж.Расулов',
                           'Конибодом', 'Дусти', 'Ниёзбек','Истаравшан', 'Рогун','Гончи', 'Чашмасор', 'Нофароч', 'Ободи',
                           'Каракчикум', 'Оббурдон', 'Куруш', 'Ворух', 'Гулякандоз', 'Некфайз', 'Сомгор', 'Пунук', 'Панчакент',
                           'Кулканд', 'Оппон', 'Файзобод', 'Турсунзода', 'Гусар', 'Равшан','Ифтихор', 'Х.Алиев', 'Ёри',
                           'Мучун', 'Саразм'])
     if region == 'Шахристон':
        Шахристон = 0
    else:
        Гули сурх = 1
    else:
        Худжанд-Центр = 2
    else:
        Спитамен = 3
    else:
        Шарк = 4
    else:
        Мархамат = 5
    else:
        Душанбе = 6
    else:
       Навкент = 7
    else:
        Кистакуз = 8
    else:
        Худжанд-Панчшанбе = 9
    else:
        Бустон = 10
    else:
        Истаравшан-филиал = 11
    else:
        Рудаки = 12
    else:
        Ашт = 13
    else:
        Калининобод = 14
    else:
        Сино = 15
    else:
        Исфара = 16
    else:
        Хисор = 17
    else:
        Зафаробод = 18
    else:
        Ничони = 19
    else:
        Вахдат = 20
    else:
        Мехнатобод = 21
    else:
       Уяс = 22
    else:
        Дж.Расулов = 23
    else:
        Конибодом = 24
    else:
        Дусти = 25
    else:
        Ниёзбек = 26
    else:
        Истаравшан = 27
    else:
        Рогун = 28
    else:
        Гончи = 29
    else:
        Чашмасор = 30
    else:
        Нофароч = 31
    else:
        Ободи = 32
    else:
        Каракчикум = 33
    else:
        Оббурдон = 34
    else:
        Куруш = 35
    else:
        Ворух = 36
    else:
       Гулякандоз = 37
    else:
        Некфайз = 38
    else:
        Сомгор = 39
    else:
        Пунук = 40
    else:
        Панчакент = 41
    else:
        Кулканд = 42
    else:
        Оппон = 43
    else:
        Файзобод = 44
    else:
        Турсунзода = 45
    else:
        Гусар = 46
    else:
        Равшан = 47
    else:
        Ифтихор = 48
    else:
        Х.Алиев = 49
    else:
        Ёри = 50
    else:
        Мучун = 51
    else:
       Саразм = 52


    Direction_of_activity = st.radio("Укажите ваше направление деятельности:", ['Животноводство и переработка молока', 'Приобретение техники',
                                     'Ремонт дома', 'торговля', 'Земледелие', 'Приобретение мебели',
                                     'Оплата на лечение', 'Проведение мероприятий', 'Оплата поездок',
                                     'Услуги', 'Переоборудование транспорта', 'Потребнужды',
                                     'Оплата образования', 'Производство', 'Покупка квартиры',
                                     'Потреб.другое', 'Ремонт места деятельности', 'Сельское хозяйство',
                                     'Все', 'Сушка фруктов', 'Коммерческий'])
    if direction == 'Животноводство и переработка молока':
        Чорводори ва коркарди шир = 0
    else:
        Приобретение техники = 1
    else:
        Ремонт дома = 2
    else:
        торговля = 3
    else:
        Земледелие = 4
    else:
        Приобретение мебели = 5
    else:
        Оплата на лечение = 6
    else:
        Проведение мероприятий = 7
    else:
        Оплата поездок = 8
    else:
        Услуги = 9
    else:
        Переоборудование транспорта = 10
    else:
        Потребнужды = 11
    else:
        Оплата образования = 12
    else:
        Производство = 13
    else:
        Покупка квартиры = 14
    else:
        Потреб.другое = 15
    else:
        Ремонт места деятельности = 16
    else:
        Сельское хозяйство = 17
    else:
        Все = 18
    else:
        Сушка фруктов = 19
    else:
        Коммерческий = 20
    
     Currency_code = st.radio("В какой валюте вы бы хотели получить кредит:", ['Доллар США', 'Сомони', Рос.рубль])
    if currency == 'Мужской':
        Доллар США = 0
    else:
        Сомони = 1
    else:
        Рос.рубль = 2


     Pledge_code = st.radio("В какой валюте вы бы хотели получить кредит:", ['Группа', 'Категория 1', 'Категория 2', 'Категория 3', 'Категория 4'])
    if Pledge == 'Группа':
        Группа = 0
    else:
        Категория 1 = 1
    else:
        Категория 2 = 2
    else:
        Категория 3 = 3
    else:
        Категория 4 = 4

 
    
    
    
    educ =  ['Аспирантура', 'Оли', 'Миёнаи махсус', 'Олии нопурра', 'Миёна', 'Миёнаи нопурра']
    education = st.selectbox('Уровень образования:', educ)
    label_encoder = LabelEncoder()
    encoded_educ = label_encoder.fit_transform(educ)
    Education = encoded_educ[educ.index(education)]

    
    


        
    options = ['0-5', '5-10', '10-15', '15-20', '20-25', '25-40', '40-50', '50+']
    bus_exp =  st.radio("Стаж работы:", options)
    label_encoder = LabelEncoder()
    encoded_busEx = label_encoder.fit_transform(options)
    business_experience = encoded_busEx[options.index(bus_exp)]

    Net_profit = st.number_input('Доход (в месяц):', step=1, value=0)
        
    result1 = ""
    result2 = ""
    result3 = ""
    result4 = ""
    if st.button("Predict"):
        result1, result2 = issue_a_loan(FamilySize, Loan_Amount, Loan_Term,
                 Monthly_repayment_amount_according_to_schedule, Grace_period,
                 Capital, Asset, Days_overdue, Number_of_overdue,
                 Lending_stage, Gross_profit,
                 Net_Profit, Age, Married_encoded, isFemale, nationality_encoded, educ, Filial_code,
                 Region_code, Direction_of_activity, Currency_code, Pledge_code,
                 business_experience)
        if result1 == 0:
            result3 = Credit_sum(FamilySize, Loan_Amount, Loan_Term,
                 Monthly_repayment_amount_according_to_schedule, Grace_period,
                 Capital, Asset, Days_overdue, Number_of_overdue,
                 Lending_stage, Gross_profit,
                 Net_Profit, Age, Married_encoded, isFemale, nationality_encoded, educ, Filial_code,
                 Region_code, Direction_of_activity, Currency_code, Pledge_code,
                 business_experience, result1)
            st.success(f'Сумма вам не доступна')
            st.success(f'Максимально доступная сумма: {result3}')
        else:
            st.success(f'Вероятность выдачи кредита {result1[0]*100:.2f}%')
            st.success(f'Вероятность возврата кредита вовремя: {result2[0][0] * 100:.2f}%')
            result4 = Delays_days(FamilySize, Loan_Amount, Loan_Term,
                 Monthly_repayment_amount_according_to_schedule, Grace_period,
                 Capital, Asset, Days_overdue, Number_of_overdue,
                 Lending_stage, Gross_profit,
                 Net_Profit, Age, Married_encoded, isFemale, nationality_encoded, educ, Filial_code,
                 Region_code, Direction_of_activity, Currency_code, Pledge_code,
                 business_experience)
            st.success(f'Примерная просрочка: {(result4[0]).astype(int)}')
                                                   
if __name__ == '__main__':
    main()
