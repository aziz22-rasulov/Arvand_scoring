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

    
def issue_a_loan(FamilySize, Loan Amount, Loan Term,
                 Monthly repayment amount according to schedule, Grace period (month),
                 Capital, Asset, Days overdue, Number of overdue,
                 Lending stage (which time a loan is received, Gross profit,
                 Net Profit, Age, isFemale, Filial_code,
                 Region_code, Direction of activity, Currency_code, Pledge_code,
                 business_experience):
    # Преобразование business_experience в числовой формат
    business_experience = int(business_experience)

    # Преобразование всех переменных в числовой формат
    input_data = [
        FamilySize, Loan_Amount, Loan_Term,
        Monthly_repayment_amount_according_to_schedule, Grace_period_(month),
        Capital, Asset, Days_overdue, Number_of_overdue,
        Lending_stage_(which_time_a_loan_is_received), Gross_profit,
        Net_Profit, Age, isFemale, Filial_code,
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
                 Monthly_repayment_amount_according_to_schedule, Grace_period_(month),
                 Capital, Asset, Days_overdue, Number_of_overdue,
                 Lending_stage_(which_time_a_loan_is_received), Gross_profit,
                 Net_Profit, Age, isFemale, Filial_code,
                 Region_code, Direction_of_activity, Currency_code, Pledge_code,
                 business_experience):
    # Исправлено: добавлено преобразование business_experience в числовой формат
    business_experience = int(business_experience)

    input_data = [
        FamilySize, Loan_Amount, Loan_Term,
        Monthly_repayment_amount_according_to_schedule, Grace_period_(month),
        Capital, Asset, Days_overdue, Number_of_overdue,
        Lending_stage_(which_time_a_loan_is_received), Gross_profit,
        Net_Profit, Age, isFemale, Filial_code,
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
        Monthly_repayment_amount_according_to_schedule, Grace_period_(month),
        Capital, Asset, Days_overdue, Number_of_overdue,
        Lending_stage_(which_time_a_loan_is_received), Gross_profit,
        Net_Profit, Age, isFemale, Filial_code,
        Region_code, Direction_of_activity, Currency_code, Pledge_code,
        business_experience):
    # Исправлено: добавлено преобразование business_experience в числовой формат
    business_experience = int(business_experience)

    input_data = [
       FamilySize, Loan_Amount, Loan_Term,
        Monthly_repayment_amount_according_to_schedule, Grace_period_(month),
        Capital, Asset, Days_overdue, Number_of_overdue,
        Lending_stage_(which_time_a_loan_is_received), Gross_profit,
        Net_Profit, Age, isFemale, Filial_code,
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
    Grace_period_(month) = st.number_input('Льготный период?', step=1, value=0)
    Capital = st.number_input('Ваш начальный капитал?', step=1, value=0)
    Asset = st.number_input('Ваш актив?', step=1, value=0)
    Days_overdue = st.number_input('Дни просрочки?', step=1, value=0)
    Number_of_overdue = st.number_input('Дни количество дней просрочки?', step=1, value=0)
    Lending_stage_(which_time_a_loan_is_received) = st.number_input('Какой раз вы уже получаете кредит?', step=1, value=0)
    Gross_profit = st.number_input('Валовая прибыль', step=1, value=0)
    Net_Profit = st.number_input('Чистая прибыль', step=1, value=0)
    Age = st.number_input('Сколько вам лет? запишите ваш год рождения в формате ', step=1, value=0)
    
    isFemale = st.radio("Укажите свой пол:", ['Мужской', 'Женский'])
    if sex == 'Мужской':
        Мужской = 0
    else:
        Женский = 1

    Filial_code = st.radio("Укажите филиал банка, в котром вы получаете кредит:", ['Мужской', 'Женский'])
    if filial == 'Истаравшан':
        Истаравшан = 0
    else:
        Хучанд = 1
    else:
        Ч. Расулов = 2
    else:
        Душанбе = 3
    else:
        Исфара = 4
    else:
        Панчакент = 5
        # Создаем dummy-столбцы со значениями False для всех категорий
    nationality_encoded = pd.get_dummies(data1, prefix='', prefix_sep='').astype(bool)
    # Получаем выбор пользователя
    selected_nationality = st.selectbox('Введите национальность:', data1)
    # Если выбор сделан, обновляем соответствующий dummy-столбец в True
    if selected_nationality is not None:
        nationality_encoded = nationality_encoded[selected_nationality].astype(bool)
    
    Age = st.number_input('Сколько вам полных лет?', step=1, value=0)

    family_options = ['Оиладор', 'Беоила', 'Бевамард (бевазан)', 'Чудошуда']
    familyst = st.selectbox('Семейное положение:', family_options)
    label_encoder = LabelEncoder()
    encoded_family = label_encoder.fit_transform(family_options)
    FamilyStatus = encoded_family[family_options.index(familyst)]
    
    FamilySize = st.number_input('Сколько человек в семье?', step=1, value=0)
    
    educ =  ['Аспирантура', 'Оли', 'Миёнаи махсус', 'Олии нопурра', 'Миёна', 'Миёнаи нопурра']
    education = st.selectbox('Уровень образования:', educ)
    label_encoder = LabelEncoder()
    encoded_educ = label_encoder.fit_transform(educ)
    Education = encoded_educ[educ.index(education)]

    type = st.selectbox('Тип кредита:', ['Потребительский кредит','Кредит на предпринимательскую деятельность'])
    if type=='Потребительский кредит':
        type_of_credit = 0
    else:
        type_of_credit = 1
        
    filial_encoded = pd.get_dummies(data2, prefix='', prefix_sep='').astype(bool)
    selected_filial = st.selectbox('Филиал Банка:', data2)
    if selected_filial is not None:
        filial_encoded = filial_encoded[selected_filial].astype(bool)

    

    Loan_amount = st.number_input('На какую сумму хотите взять кредит?', step=1, value=0) 

    Loan_term = st.number_input('На какой срок вы хотите взять кредит(месяц)?', step=1, value=0) 

    Lending_stage = st.number_input('Сколько кредитов вы брали(с учетом этого)?', step=1, value=0)
    if Lending_stage>1:
         have_delay = st.selectbox('Есть ли у вас просрочки?', ['Да','Нет'])
         if have_delay == 'Да':
             has_overdue = 1
         else:
             has_overdue = 0
    else:
        has_overdue = 0
    Repayment = st.number_input('Ежемесячная сумма погашения:', step=1, value=0)
    
    Grace_preiod = st.number_input('Льготный период (месяц):', step=1, value=0)
    
    have_debt = st.selectbox('У вас есть долги?', ['Да','Нет'])
    if have_debt == 'Да':
        Debt = st.number_input('Введите сумму долга:', step=1, value=0)
    else:
        Debt = 0
    if Debt > 10000:
        high_debt = 1
    else:
        high_debt = 0
        
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
        result1, result2 = issue_a_loan(Gender, FamilySize, Loan_amount, Loan_term, Repayment, Grace_preiod, Debt, Lending_stage,
                 Net_profit, Age, FamilyStatus, Education, business_experience, type_of_credit, has_overdue,
                 high_debt, nationality_encoded, filial_encoded, region_encoded, loan_goal_encoded, 
                 sector_encoded, currency_encoded, pledge_encoded)
        if result1 == 0:
            result3 = Credit_sum(Gender, FamilySize, Loan_term, Repayment, Grace_preiod, Debt, Lending_stage,
                 Net_profit, Age, FamilyStatus, Education, business_experience, type_of_credit, has_overdue,
                 high_debt, nationality_encoded, filial_encoded, region_encoded, loan_goal_encoded, 
                 sector_encoded, currency_encoded, pledge_encoded, result1)
            st.success(f'Сумма вам не доступна')
            st.success(f'Максимально доступная сумма: {result3}')
        else:
            st.success(f'Вероятность выдачи кредита {result1[0]*100:.2f}%')
            st.success(f'Вероятность возврата кредита вовремя: {result2[0][0] * 100:.2f}%')
            result4 = Delays_days(Gender, FamilySize, Loan_amount, Loan_term, Repayment, Grace_preiod, Debt, Lending_stage,
                 Net_profit, Age, FamilyStatus, Education, business_experience, type_of_credit, has_overdue,
                 high_debt, nationality_encoded, filial_encoded, region_encoded, loan_goal_encoded, 
                 sector_encoded, currency_encoded, pledge_encoded)
            st.success(f'Примерная просрочка: {(result4[0]).astype(int)}')
                                                   
if __name__ == '__main__':
    main()
