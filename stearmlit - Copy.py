import streamlit as st
import pandas as pd
import  matplotlib.pyplot as plt
import  seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# تنظیمات اولیه
st.title('SMART DOCTOR')
st.markdown('''In this project, we predict 5 diseases using machine learning,
             diseases such as high blood pressure, depression, and diabetes....
            Here, we start with the disease, we explain each disease,
             from the beginning, we show the data, and then the number of patients,
             the number of people who are sick and healthy, and then we show you the histograms of each type,
             and then you have to choose your model, which are random forest and .... 
            Then we get to the amount of test and train, it is better to keep them as they are and not change them,
             and then we get to the model you chose and the accuracy of the model you chose, for example, if it was 0.90,
             it means that it predicts 90% correctly, and then we get to the clutter matrix,
             where the left side is what the machine predicted and the bottom is what is real.
            Then you have to tell the machine what it asked you to do so that the machine can predict.
            And then when you're done, click the prediction option. If the green box appears,
             it means you're healthy. If the red box appears, it means you're sick.
             There's a section below the prediction called Help for when you're sick and don't know what to do to get better.
..''')
st.warning('''A very important point: If you are asked a yes or no question, 
           you should not write yes or no.You should answer with a number.
            If your answer is yes, write 1 and if no, write 0.''')
st.warning('''If you dont understand what the machine is asking you for, 
           there is a help icon at the top of each data, which if you 
           click on will explain that data to you.''')

# بارگذاری داده (فقط یک بار با استفاده از کش)
@st.cache_data
def load_data():
    data = pd.read_csv(r'c:\Users\ASUS\OneDrive\Desktop\diabetes.csv')
    return data

# اگر دکمه Diabetes کلیک شد:
if st.sidebar.button('Diabetes'):
    st.session_state.show_diabetes = True  # ذخیره حالت نمایش

# اگر وضعیت نمایش دیابت فعال بود:
if st.session_state.get('show_diabetes', False):
    st.title('Diabetes')
    st.markdown('''Diabetes: In this section, based on data such as glucose,
                 etc., the machine predicts whether you have diabetes or not.
                 If the blue box appears, it means you are healthy, and if the پ
                red box appears, it means you are sick.''')
    st.warning('It is only for women.')
    
    df = load_data()  # بارگذاری داده
    
    # نمایش داده در صورت انتخاب چک‌باکس
    if st.checkbox('show data', key='show_data_checkbox'):
        st.subheader('raw data')
        st.write(df)
    st.write(f'Number of healthy people in the data : {len(df)}')
    st.write(f'Number of healthy patients in the data :{len(df[df['Outcome']==0])}')
    st.write(f'Number of diabetic patients in the data : {len(df[df['Outcome']==1])}')
    if st.checkbox('Do you want me to show you the histogram for each feature?'):
        feature=st.selectbox('Select a feature to display the histogram.',df.columns[:-1])
        plt.figure(figsize=(8,5))
        sns.histplot(data=df,x=feature,hue='Outcome',kde=True)
        st.pyplot(plt)
    model_name=st.selectbox('Model Selection',
                            ['Logistic Regression',
                             'Support Vector Machine',
                             'Random Forest'])
    
    test_size=st.slider('Test data share ',10,40,20,help='Give me the amount to test.')
    random_state=st.slider('Random value',10,40,20,help='Give me the amount of training.')

    x=df.drop('Outcome',axis=1)
    y=df['Outcome']
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=test_size/100,random_state=random_state)
    def get_model(model_name):
        if model_name =='Random Forest':
            model = RandomForestClassifier(random_state=random_state)
        elif  model_name==  'Support Vector Machine':
            model=SVC(probability=True,random_state=random_state)
        else:
            model=LogisticRegression(random_state=random_state)
        return model
    model=get_model(model_name)
    model.fit(x_train,y_train)

    y_pred=model.predict(x_test)
    accuracy=accuracy_score(y_test,y_pred)
    st.markdown('Model results')
    st.write(f'The model you chose : **{model_name}**')
    st.write(f'Accuracy of the model : **{accuracy:.2f}**')
    if st.checkbox('View matrices'):
        cm=confusion_matrix(y_test,y_pred)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=['                                        Healthy                           ,                      diabetic     '],
                    yticklabels=['Healthy                                  , diabetic                              '])
        st.pyplot(plt)
    if st.button('Classification report'):      
        st.markdown('Classification report')
        report=classification_report(y_test,y_pred,output_dict=True)
        report_df=pd.DataFrame(report).transpose()
        st.write(report_df)
    st.title('Predicting new data')
    st.title('Enter the values')
    pregnancies=st.text_input('Number of pregnancies',help='How many times have you been pregnant?')
    glucose=st.text_input('Number of glucose',help='Glucose is in the test.')
    blood_pressure=st.text_input('Number of blood_pressure',help='Measure your blood pressure with a blood pressure monitor.')
    skin_thicknees=st.text_input('Number of skin thicknees',help='Skin thickness (skin thickness) in tests can include:TB test, Fitzpatrick management, etc.')
    insulin=st.text_input('Number of insulin',help='insulin is in the test ')
    bmi=st.text_input('Number of BMi',help='To calculate your BMI, you need to know your weight and height and use a BMI calculator. BMI calculators are readily available online and on mobile apps.')
    dpf=st.text_input('Number of dpf',help='dpf is an abbreviation for "differential" and means counting and differentiating the types of white blood cells (WBC) in the test.')
    age=st.text_input('Number of age',help='How old are you')
    if st.button('predicting'):
        try:
        # تبدیل مقادیر به عدد (اگر خالی باشند، 0 قرار دهید)
            pregnancies = int(pregnancies) if pregnancies else 0
            glucose = int(glucose) if glucose else 0
            blood_pressure = int(blood_pressure) if blood_pressure else 0
            skin_thicknees = int(skin_thicknees) if skin_thicknees else 0
            insulin = int(insulin) if insulin else 0
            bmi = float(bmi) if bmi else 0.0  # اگر BMI اعشاری است از float استفاده کنید
            dpf = float(dpf) if dpf else 0.0
            age = int(age) if age else 0

            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thicknees, insulin, bmi, dpf, age]])
            prediction = model.predict(input_data)
            probability = model.predict_proba(input_data)[0][1]

            if prediction[0] == 1:
                st.error(f'Result: Diabetic (Probability: {probability:.2%})')
            else:
                st.success(f'Result: Healthy (Probability: {(1 - probability):.2%})')

        except ValueError:  # اگر کاربر عدد وارد نکند (مثلاً حروف بنویسد)
            st.error("لطفاً فقط عدد وارد کنید!")
        except Exception as e:  # سایر خطاهای غیرمنتظره
            st.error(f"خطایی رخ داد: {e}")
    st.title('DIABET HELP')

    if st.button('HELP'):
        st.title('Guide to treating diabetes')
        st.markdown('''To manage and control diabetes, lifestyle measures and drug treatments are necessary. 
                    These measures include following a healthy diet, exercising regularly, managing stress, 
                    and taking medications prescribed by your doctor if necessary. Regular blood sugar monitoring 
                    and consulting with your doctor to develop a treatment plan are also essential.Steps to Manage
                    Diabetes : ''')
        st.markdown('___Healthy diet___ : ')
        st.write('''Eating complex carbohydrates (like whole grains) instead of simple carbohydrates (like white bread).
                    Increasing fiber intake through fruits and vegetables. Choosing healthy protein sources like fish, 
                    eggs, and legumes. Consuming healthy fats like olive oil and avocado. Reducing sugar and saturated fat intake. 
    ''')
        st.markdown('___Regular exercise___ :  ')
        st.markdown('''Do aerobic activity, such as brisk walking, for 150 minutes a week.
                    Do lighter exercises, such as yoga or tai chi. Regular physical activity
                    to lower blood sugar and improve insulin sensitivity.''')
        st.markdown('___Stress management___ : ')
        st.markdown('''Practice relaxation techniques such as meditation and deep breathing. 
                    Reduce stress with relaxing activities and getting enough sleep.''')
        st.markdown('___Medication___ : ')
        st.markdown('''Use oral medications or insulin as prescribed by your doctor. 
                    Regular blood sugar monitoring to adjust medication dosage.''')
        st.markdown('___Blood sugar monitoring___ : ')
        st.markdown('''Use blood sugar monitoring devices to measure blood sugar regularly.
                    Consult your doctor to set up a treatment plan and adjust your medication dosage.''')
        st.markdown('___Quit smoking___ : ')
        st.markdown('Smoking increases the risk of diabetes and impairs blood sugar control.')
    
        st.markdown('___Reduce alcohol consumption___ : ')
        st.markdown('Alcohol can increase blood sugar and interfere with weight loss efforts.')

        st.warning('___Important points___ :')

        st.markdown('___Consult a doctor___ : ')
        st.markdown('Always consult your doctor to set up a treatment plan and take medications.')

        st.markdown('___Learning about diabetes___ : ')
        st.markdown('Diabetes self-management education can help you better manage your condition.')

        st.markdown('___The importance of sleep___ : ')
        st.markdown('''Adequate and quality sleep is essential for regulating 
                    metabolism and hormones, and thus for better diabetes control.''')
        st.markdown('___Weight loss___ : ')
        st.markdown(' Losing weight if you are overweight can help control blood sugar.')

        st.markdown('___Have a regular meal plan___ : ')
        st.markdown(' Divide meals into three main meals and avoid high-calorie snacks.')
    
        st.success('''By following these tips and having a healthy lifestyle
                   you can manage diabetes well and prevent its complications.''')
    
        st.warning('''This content is for informational purposes only.
                    For medical advice or diagnosis, please consult a
                    professional. AI responses may contain errors.''')
    
        
@st.cache_data
def Heart():
    data = pd.read_csv(r'C:\Users\ASUS\Downloads\heart.csv')
    return data

# اگر دکمه Diabetes کلیک شد:
if st.sidebar.button('Heart disease'):
    st.session_state.show_diabetes = True  # ذخیره حالت نمایش

# اگر وضعیت نمایش دیابت فعال بود:
if st.session_state.get('show_diabetes', False):
    st.title('Heart disease')
    st.markdown('''Heart disease is predicted by chol, ..... 
                If you had heart disease and didn't know what to do,
                go to the heart disease help section..''')
    
    heart = Heart()  # بارگذاری داده
    
    # نمایش داده در صورت انتخاب چک‌باکس
    if st.checkbox('show data'):
        st.subheader('raw data')
        st.write(heart)
    st.write(f'Number of healthy people in the data : {len(heart)}')
    st.write(f'Number of healthy patients in the data :{len(heart[heart['output']==0])}')
    st.write(f'Number of heart patients in the data : {len(heart[heart['output']==1])}')
    if st.checkbox('Do you want me to show you the histogram for each Feature?'):
        feature=st.selectbox('Select a feature to display the histogram.',heart.columns[:-1])
        plt.figure(figsize=(8,5))
        sns.histplot(data=heart,x=feature,hue='output',kde=True)
        st.pyplot(plt)
    model_name=st.selectbox('Model selection',
                            ['Logistic Regression',
                             'Support Vector Machine',
                             'Random Forest'])
    
    test_size=st.slider('Test Data share ',10,40,20,help='Give me the amount to Test.')
    random_state=st.slider('Random Value',10,40,20,help='Give me the amount of Training.')

    x=heart.drop('output',axis=1)
    y=heart['output']
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=test_size/100,random_state=random_state)
    def get_model(model_name):
        if model_name =='Random Forest':
            model = RandomForestClassifier(random_state=random_state)
        elif  model_name==  'Support Vector Machine':
            model=SVC(probability=True,random_state=random_state)
        else:
            model=LogisticRegression(random_state=random_state)
        return model
    model=get_model(model_name)
    model.fit(x_train,y_train)

    y_pred=model.predict(x_test)
    accuracy=accuracy_score(y_test,y_pred)
    st.markdown('Model results')
    st.write(f'The model you chose : **{model_name}**')
    st.write(f'Accuracy of the model : **{accuracy:.2f}**')
    if st.checkbox('View Matrices'):
        cm=confusion_matrix(y_test,y_pred)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=['                                       Healthy                           ,                      heart diases     '],
                    yticklabels=['Healthy                                  , heart diaeses                              '])
        st.pyplot(plt)
    if st.button('Classification Report'):      
        st.markdown('Classification Report')
        report=classification_report(y_test,y_pred,output_dict=True)
        report_df=pd.DataFrame(report).transpose()
        st.write(report_df)
    st.title('Predicting new data')
    st.title('Enter the values')
    age=st.text_input('Number of Age',help='How old are you?')
    sex=st.text_input('Number of sex',help='Please indicate your gender. If you are female, enter 0 and if you are male, enter 1.')
    cp=st.text_input('Number of Cp',help='cp is present in a blood test. Cp in a blood test can refer to Ceruloplasmin')
    trtbps=st.text_input('Number of Trtbps',help='Systolic blood pressure is the top number on a blood pressure reading (for example, 120 for a blood pressure of 120/80 mmHg). This number represents the pressure exerted on the walls of the arteries when the heart contracts and pumps blood into the arteries.')
    chol=st.text_input('Number of Chol',help='Cholesterol (chol) is in the test.')
    fbs=st.text_input('Number of Fbs',help='FBS is present in blood testsFBS is present in blood tests and stands for Fasting Blood Sugar')
    restecg=st.text_input('Number of Restecg',help='restecg means resting ECG, which is divided into 0 (normal), 1 (mild abnormalities), and 2 (clear abnormalities).')
    thalachh=st.text_input('Number of Thalachh',help='Thalachh This parameter indicates the patients highest heart rate during exercise or stress. If you havent taken this test, subtract your age from 220, but its better if you do.')
    exng=st.text_input('Number of Exng',help='This center shows whether someone who took the exercise test felt chest pain or not.')
    slp=st.text_input('Number of Slp',help='SLP on the ECG indicates whether the person has ST segment elevation or not. 0 indicates coronary disease, 1 means flat elevation, and 2 means ascending elevation.')
    oldpeak=st.text_input('Number of Oldpeak',help='Old peak indicates the degree of drop or depression of the ST segment of the ECG after exercise testing.')
    caa=st.text_input('Number of Caa',help='The number of primary colors seen in angiography is taken.')
    thall=st.text_input('Number of Thall',help='thall indicates thallium test results (thallium stress test)')
    if st.button('Predicting'):
        try:
            # تبدیل مقادیر به عدد (اگر خالی باشند، 0 قرار دهید)
            pregnancies = int(pregnancies) if pregnancies else 0
            glucose = int(glucose) if glucose else 0
            blood_pressure = int(blood_pressure) if blood_pressure else 0
            skin_thicknees = int(skin_thicknees) if skin_thicknees else 0
            insulin = int(insulin) if insulin else 0
            bmi = float(bmi) if bmi else 0.0  # اگر BMI اعشاری است از float استفاده کنید
            dpf = float(dpf) if dpf else 0.0
            age = int(age) if age else 0
        

            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thicknees, insulin, bmi, dpf, age]])
            prediction = model.predict(input_data)
            probability = model.predict_proba(input_data)[0][1]

            if prediction[0] == 1:
                st.error(f'Result: Diabetic (Probability: {probability:.2%})')
            else:
                st.success(f'Result: Healthy (Probability: {(1 - probability):.2%})')

        except ValueError:  # اگر کاربر عدد وارد نکند (مثلاً حروف بنویسد)
            st.error("لطفاً فقط عدد وارد کنید!")
        except Exception as e:  # سایر خطاهای غیرمنتظره
            st.error(f"خطایی رخ داد: {e}")
    st.title('Heart dieases Help')
    if st.button('Help'):
        st.title('Heart disease treatment guide')
        st.markdown('''Treatment for heart disease depends on the type and 
                    severity of the disease, as well as the patients individual 
                    circumstances. In general, treatment options include 
                    lifestyle changes, drug therapy, minimally invasive
                     interventions (such as angioplasty and stenting),
                     and cardiovascular surgery.''')
        st.markdown('Treatment methods for heart diseases : ')

        st.markdown('___Lifestyle changes___:')
        st.markdown('''This includes things like a healthy diet, 
                    regular exercise, quitting smoking, reducing 
                    alcohol consumption, and managing stress.''')
        st.markdown('___Drug treatment___:')
        st.markdown('''Various medications are used to treat heart disease, 
                    including anticoagulants, cholesterol-lowering drugs,
                     blood pressure-lowering drugs, and heart rate-regulating drugs.''')
        st.markdown('___Minimally invasive interventions___:')
        st.markdown('''These procedures include angioplasty and 
                    stenting to open blocked coronary arteries.''')
        st.markdown('___Heart surgery___:')
        st.markdown('''In more severe cases, heart surgery may be needed,
                     such as coronary artery bypass surgery, heart valve 
                    repair or replacement, or heart transplant.''')
        st.markdown('___Implantation of medical devices___:')
        st.markdown('''In some cases, devices such as a pacemaker or ICD
                     (implantable defibrillator) may be implanted to regulate
                     the heart rate or prevent cardiac arrest.''')
        st.markdown('___Specific treatments for some heart diseases___ : ')

        st.markdown('___Heart failure___ :')
        st.markdown('''Treatment includes medications to strengthen the hearts function,
                    reduce the hearts workload, and control fluid retention.
                    In more severe cases, a heart transplant may be needed.''')
        st.markdown('___Coronary artery disease___ : ')
        st.markdown('''Treatment includes lifestyle changes, medications,
                     angioplasty and stenting, or bypass surgery.''')
        st.markdown('___Cardiac arrhythmia___ : ')
        st.markdown('''Treatment includes medication, ablation 
                    (burning away abnormal heart tissue), or implantation of pacemakers.''')
        st.markdown('___Heart valve diseases___ : ')
        st.markdown('Treatment includes medication, surgical repair, or replacement of the heart valve.')

        st.warning('___Important points___ :  ')
        st.markdown('''The choice of the appropriate treatment method for each patient
                     should be made in consultation with a cardiologist.It is important
                     for patients to be monitored regularly by their doctor and,
                     if necessary, to change their treatment plan according to
                    new conditions.Home remedies may be helpful in managing
                     symptoms and preventing worsening of the disease, but
                     they should not be used as a substitute for medical treatments.''')
        st.warning('''This content is for informational purposes only.
                    For medical advice or diagnosis, please consult a
                    professional. AI responses may contain errors.''')
 



@st.cache_data
def P():
    data = pd.read_csv(r'C:\Users\ASUS\Downloads\heart_failure_clinical_records_dataset.csv')
    return data

# اگر دکمه Diabetes کلیک شد:
if st.sidebar.button('Heart failure'):
    st.session_state.show_diabetes = True  # ذخیره حالت نمایش

# اگر وضعیت نمایش دیابت فعال بود:
if st.session_state.get('show_diabetes', False):
    st.title('Heart failure')
    st.markdown('''Heart failure is predicted based on anemia, etc.
                 If you don't understand something, press the help button.''')
    
    p = P()  # بارگذاری داده
    
    # نمایش داده در صورت انتخاب چک‌باکس
    if st.checkbox('Show data'):
        st.subheader('raw data')
        st.write(heart)
    st.write(f'Number of healthy people in the data : {len(p)}')
    st.write(f'Number of healthy patients in the data :{len(p[p['DEATH_EVENT']==0])}')
    st.write(f'Number of heart patients in the data : {len(p[p['DEATH_EVENT']==1])}')
    if st.checkbox('Do You want me to show you the histogram for each Feature?'):
        feature=st.selectbox('Select a feature to display the histogram.',p.columns[:-1])
        plt.figure(figsize=(8,5))
        sns.histplot(data=p,x=feature,hue='DEATH_EVENT',kde=True)
        st.pyplot(plt)
    model_name=st.selectbox('MODEL Selection',
                            ['Logistic Regression',
                             'Support Vector Machine',
                             'Random Forest'])
    
    test_size=st.slider('Test Data share ',10,40,20,help='Give Me the amount to Test.')
    random_state=st.slider('Random Value',10,40,20,help='Give Me the amount of Training.')

    x=p.drop('DEATH_EVENT',axis=1)
    y=p['DEATH_EVENT']
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=test_size/100,random_state=random_state)
    def get_model(model_name):
        if model_name =='Random Forest':
            model = RandomForestClassifier(random_state=random_state)
        elif  model_name==  'Support Vector Machine':
            model=SVC(probability=True,random_state=random_state)
        else:
            model=LogisticRegression(random_state=random_state)
        return model
    model=get_model(model_name)
    model.fit(x_train,y_train)

    y_pred=model.predict(x_test)
    accuracy=accuracy_score(y_test,y_pred)
    st.markdown('Model results')
    st.write(f'The model you chose : **{model_name}**')
    st.write(f'Accuracy of the model : **{accuracy:.2f}**')
    if st.checkbox('VIew Matrices'):
        cm=confusion_matrix(y_test,y_pred)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=['                                       Healthy                           ,                      heart failure     '],
                    yticklabels=['Healthy                                  , heart failure                              '])
        st.pyplot(plt)
    if st.button('CLassification Report'):      
        st.markdown('CLassification Report')
        report=classification_report(y_test,y_pred,output_dict=True)
        report_df=pd.DataFrame(report).transpose()
        st.write(report_df)
    st.title('Predicting new data')
    st.title('Enter the values')
    age=st.text_input('Number Of Age',help='How old are you')
    anaemia=st.text_input('Number of anaemia',help='If you are not anemic, select option 0 and if you are anemic, select option 1.')
    creatinine_phosphokinase=st.text_input('Number of creatinine_phosphokinase',help='Please indicate the size of the creatinine phosphokinase test. If you did not take the test, enter 0.')
    diabetes=st.text_input('Number of diabetes',help='If you do not have diabetes, enter 0, if you do, enter 1, and if you have not been tested for diabetes, take the test on this site in the diabetes section.')
    ejection_fraction=st.text_input('Number of ejection_fraction',help='It is an important indicator of heart function, indicating what percentage of the blood in the left ventricle is pumped with each heartbeat. It is determined in an echocardiogram.')
    high_blood_pressure=st.text_input('Number of high_blood_pressure',help='If your blood pressure is mostly high, enter 1, and if you do not have high blood pressure, enter 0.')
    platelets=st.text_input('Number of Blood platelets',help='Platelets, or thrombocytes, are present in a blood test and are measured as part of a complete blood count (CBC) test.')
    serum_creatinine=st.text_input('Number of serum_creatinine',help='Serum creatinine is present in the blood test.')
    serum_sodium=st.text_input('Number of serum_sodium',help='Sodium (Na) is present in the blood test.')
    st.warning('If you are female, select option 0 and if you are male, select option 1.')
    sex=st.text_input('Number of sex',help='If you are female, select option 0 and if you are male, select option 1.')
    smoking=st.text_input('Number of smoking',help='If you do not smoke, select option 0. If you smoke, select option 1.')
    time=st.text_input('Number of time',help='That is, how long have you been feeling heart failure?')
    if st.button('PREDICTING'):
        try:
            age = int(age) if age else 0
            anaemia = int(anaemia) if anaemia else 0
            creatinine_phosphokinase= int(creatinine_phosphokinase) if creatinine_phosphokinase else 0
            diabetes = int(diabetes) if diabetes else 0
            ejection_fraction = int(ejection_fraction) if ejection_fraction else 0
            high_blood_pressure = int(high_blood_pressure) if high_blood_pressure else 0
            platelets=int(platelets) if platelets else 0
            serum_creatinine = int(serum_creatinine) if serum_creatinine else 0
            serum_sodium = int(serum_sodium) if serum_sodium else 0
            sex = int(sex) if sex else 0
            smoking = int(smoking) if  smoking else 0
            time = int(time) if time else 1

            input_data = np.array([[age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets,serum_creatinine, serum_sodium,sex,smoking,time]])
            prediction = model.predict(input_data)
            probability = model.predict_proba(input_data)[0][1]

            if prediction[0] == 1:
                st.error(f'Result: Diabetic (Probability: {probability:.2%})')
            else:
                st.success(f'Result: Healthy (Probability: {(1 - probability):.2%})')

        except ValueError:  # اگر کاربر عدد وارد نکند (مثلاً حروف بنویسد)
            st.error('Please enter only numbers.')
        except Exception as e:  # سایر خطاهای غیرمنتظره
            st.error(f'An error has occurred.: {e}')
    st.title('Help Heart failure')
    if st.button('heart failure treatment guide'):
        st.markdown('''Treatment for heart failure involves a 
                    combination of lifestyle changes, medications,
                    medical devices, and, in severe cases, surgery.
                    The goal of treatment is to improve heart function, 
                    reduce complications, and prevent the disease from progressing.''')
        st.markdown('___Non-drug treatments___ : ')
        st.markdown('___Lifestyle changes___:')
        st.markdown('''It includes a low-salt diet, regular physical activity
                    (in consultation with a doctor), weight loss, stress management,
                    smoking cessation, and limiting alcohol consumption.''')
        st.markdown('___Healthy diet___ : ')
        st.markdown('''Reduce salt intake, increase fruit and vegetable intake,
                     consume healthy fats, and control sugar and carbohydrate intake.''')
        st.markdown('___Drug treatments___ : ')
        st.markdown('___Diuretics___ : These medications help reduce fluid retention in the body and reduce swelling.')
        st.markdown('___Angiotensin-converting enzyme inhibitors (ACE inhibitors)___ : Help dilate blood vessels and lower blood pressure.')
        st.markdown('___Angiotensin receptor blockers (ARBs)___ : An alternative to ACE inhibitors in patients who cannot tolerate them.')
        st.markdown('___Beta-blockers___ : Lower blood pressure and heart rate.')
        st.markdown('___Cardiac glycosides (such as digoxin)___ : help strengthen the contraction of the heart.')
        st.markdown('___Anticoagulants___ : These are prescribed to prevent blood clots.')
        st.markdown('___Medical devices___ : ')
        st.markdown('___Pacemakers___ : To regulate the heart rate in cases where the heart beats irregularly.')
        st.markdown('___ICD devices___ : To prevent dangerous heart arrhythmias.')
        st.markdown('___Cardiac resynchronization therapy (CRT)___ : To improve the coordination of contraction of the hearts ventricles.')
        st.markdown('___Surgery___ :')
        st.markdown('___Coronary artery bypass surgery___ : To improve blood supply to the heart.')
        st.markdown('___Heart valve repair or replacement___ : To correct valve problems that have caused heart failure.')
        st.markdown('___Heart transplant___ : in severe and incurable cases.')
        st.warning('Important points:')
        st.markdown('''Heart failure treatment should be done under the supervision of 
                    a cardiologist.It is very important to take your medications regularly
                    and follow your doctor's recommendations.If you develop any new symptoms
                    or if your previous symptoms worsen, you should see a doctor immediately.''')
        st.warning('This content is for informational purposes only. For medical advice or diagnosis, please consult a professional. AI responses may contain errors.')
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# دکمه سایدبار برای فعال‌سازی مدل
if st.sidebar.button('depressed'):
    st.session_state.show_depression = True  # حالت نمایش را فعال می‌کند

# اگر دکمه سایدبار زده شد، این بخش اجرا می‌شود
if st.session_state.get('show_depression', False):
    st.title('Heart Failure Prediction')
    st.markdown('''Prediction of depression based on text input.''')

    # داده‌های آموزشی
    a = [
        ('Im happy', 'not depressed'),
        ('Im like it', 'not depressed'),
        ('Im love you', 'not depressed'),
        ('Im sad', 'depressed'),
        ('Im very happy', 'not depressed'),
        ('Im beutuful', 'not depressed'),
        ('Im very sad', 'depressed'),
        ('Im like life', 'not depressed'),
        ('Im not like life', 'depressed'),
        ('Im bad', 'depressed'),
        ('Be happy. Be confident. Be kind', 'not depressed'),
        ('gereck awe thanks 😊', 'not depressed'),
        ('I enjoy it every time I see it.', 'not depressed'),
        ('Oh, how I love Love Love Gully who is getting Billy Joels numbers!!!', 'not depressed'),
        ('I have a hard life', 'depressed'),
        ('I dont like myself.', 'depressed'),
        ('I love myself, my life, my home, and my school very much.', 'not depressed'),
        ('I dont like myself, my life, my home, and my school at all.', 'depressed'),
        ('And now that Im experiencing the opposite, Im very upset and sad.', 'depressed'),
        ('I have a very sad life.', 'depressed'),
        ('I have a very happy life', 'not depressed'),
        ('I dont like myself.', 'depressed'),
        ('I like myself', 'not depressed'),
        ('I started my LPN program in October this year.', 'not depressed'),
        ('I dont want to do anything because I have nothing to do right now. I dont feel well.', 'not depressed'),
        ('Im seriously thinking about taking a day off to just stream all day, I really need it 😂', 'not depressed'),
        ('I like to cry.', 'depressed')
    ]

    # تبدیل به دیتافریم
    f = pd.DataFrame(a, columns=['text', 'sentiment'])

    # مدل‌سازی
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(f['text'])
    y = f['sentiment']
    model = MultinomialNB()
    model.fit(X, y)

    # ورودی کاربر و پیش‌بینی
    user_input = st.text_input('Talk to me for a while:')
    if st.button('Predict'):  # دکمه پیش‌بینی
        if user_input:
            input_vector = vectorizer.transform([user_input])
            prediction = model.predict(input_vector)[0]
            if prediction == 'depressed':
                st.error('You seem depressed. ❤️ Please take care.')
            else:
                st.success('You seem happy! 😊')
        else:
            st.warning('Please enter some text!')
    st.title('Help depression')
    if st.button('depression treatment guide'):
        st.markdown('''Treatment for depression usually involves a
                     combination of different approaches, including
                     psychotherapy, medication, and lifestyle changes.
                     The goals of these approaches are to reduce symptoms, 
                    improve a persons functioning, and prevent the depression 
                    from returning.''')
        st.markdown('___The main methods of treating depression___ :')
        st.markdown('___Psychotherapy (talk therapy)___ :')
        st.markdown('''It includes cognitive behavioral therapy 
                    (CBT) and interpersonal therapy, which helps a
                     person identify and change negative thought and 
                    behavioral patterns and learn better coping skills.''')
        st.markdown('___Drug therapy___ :')
        st.markdown('Antidepressants help improve symptoms of depression by regulating brain chemicals.')
        
        st.markdown('___Lifestyle changes___ :')
        st.markdown('This includes regular physical activity, eating a healthy diet, regulating sleep, and reducing stress.')

        st.markdown('___Other methods___:')
        
        st.markdown('___Transcranial magnetic stimulation (rTMS)___ :')
        st.markdown('This non-invasive procedure is used to treat severe depression that does not respond to other treatments.')
        
        st.markdown('___Transcranial electrical stimulation (tDCS)___:')
        st.markdown('This method also uses electrical current to stimulate the brain and improve symptoms of depression.')
        
        st.markdown('___Important points___ : ')
        st.markdown('''Depression treatment should be done by a specialist.A combination
                     of different treatment methods is usually more effective.Patience and
                     consistent follow-up are essential in treating depression.Individual
                     measures such as exercise, healthy eating, and sleep regulation can be
                     effective alongside specialist treatments.''')
        st.warning('This content is for informational purposes only. For medical advice or diagnosis, please consult a professional. AI responses may contain errors.')




@st.cache_data
def A():
    data = pd.read_csv(r'C:\Users\ASUS\Downloads\framingham.csv')
    return data

# اگر دکمه Diabetes کلیک شد:
if st.sidebar.button('high blood pressure'):
    st.session_state.show_diabetes = True  # ذخیره حالت نمایش

# اگر وضعیت نمایش دیابت فعال بود:
if st.session_state.get('show_diabetes', False):
    st.title('high blood pressure')
    st.markdown('''High blood pressure is predicted by things 
                like diabetes, and if you don't understand something, 
                hit the help button.''')

    
    
    a = A()  # بارگذاری داده
    a.fillna(a.mean(), inplace=True)

# یا برای ستون‌های غیرعددی
# a.fillna(method='ffill', inplace=True)  # بارگذاری داده
    
    # نمایش داده در صورت انتخاب چک‌باکس
    if st.checkbox('Show Data'):
        st.subheader('raw data')
        st.write(a)
    st.write(f'Number of healthy people in the data : {len(a)}')
    st.write(f'Number of healthy patients in the data :{len(a[a['Risk']==0])}')
    st.write(f'Number of heart patients in the data : {len(a[a['Risk']==1])}')
    if st.checkbox('Do You Want me to show you the histogram for each Feature?'):
        feature=st.selectbox('Select a feature to display the histogram.',p.columns[:-1])
        plt.figure(figsize=(8,5))
        sns.histplot(data=a,x=feature,hue='Risk',kde=True)
        st.pyplot(plt)
    model_name=st.selectbox('MODEL SELECTION',
                            ['Logistic Regression',
                             'Support Vector Machine',
                             'Random Forest'])
    
    test_size=st.slider('Test Data share ',10,40,20,help='Give Me The amount to Test.')
    random_state=st.slider('Random Value',10,40,20,help='Give Me The amount of Training.')

    x=a.drop('Risk',axis=1)
    y=a['Risk']
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=test_size/100,random_state=random_state)
    def get_model(model_name):
        if model_name =='Random Forest':
            model = RandomForestClassifier(random_state=random_state)
        elif  model_name==  'Support Vector Machine':
            model=SVC(probability=True,random_state=random_state)
        else:
            model=LogisticRegression(random_state=random_state)
        return model
    model=get_model(model_name)
    model.fit(x_train,y_train)

    y_pred=model.predict(x_test)
    accuracy=accuracy_score(y_test,y_pred)
    st.markdown('Model results')
    st.write(f'The model you chose : **{model_name}**')
    st.write(f'Accuracy of the model : **{accuracy:.2f}**')
    if st.checkbox('VIEW Matrices'):
        cm=confusion_matrix(y_test,y_pred)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=['                                       Healthy                           ,                      high blood pressure     '],
                    yticklabels=['Healthy                                  , high blood pressure                              '])
        st.pyplot(plt)
    if st.button('CLASSIFICATION Report'):      
        st.markdown('CLASSIFICATION Report')
        report=classification_report(y_test,y_pred,output_dict=True)
        report_df=pd.DataFrame(report).transpose()
        st.write(report_df)
    st.title('Predicting new data')
    st.title('Enter the values')
    male=st.text_input('Number Of male',help='If you are male, enter 1 and if you are female, enter 0.')
    age=st.text_input('Number of age',help='How Old are you')
    currentSmoker=st.text_input('Do you smoke?',help='If you smoke, option 1, and if you dont smoke, option 0.')
    cigsPerDay=st.text_input('How many cigarettes do you smoke in a day?',help='How many cigarettes do you smoke in a day?')
    BPMeds=st.text_input('Do you use BPMeds drug ?',help='Blood pressure medications (BP Meds) are used to lower high blood pressure and are usually prescribed when lifestyle changes alone are not enough to control blood pressure. These medications work by mechanisms to bring blood pressure back into the normal range.For example, diuretics . Rate 1 if you use them and 0 if you dont.')
    diabetes=st.text_input('Do you have diabetes?',help='If you have diabetes, enter 1 and if you do not have diabetes, enter 0.')
    totChol=st.text_input('Number of Blood totChol',help='Total Cholesterol (Total Chol) is actually the total amount of cholesterol in the blood test.')
    sysBP=st.text_input('Number of sysBP',help='There is sysBP in your test.')
    diaBP=st.text_input('Tell me your fasting blood sugar level.',help='It is in your test.')
    BMI=st.text_input('Number of BMI',help='BMI stands for "Body Mass Index." This index is a measure of body fat based on a persons height and weight. Tell me your BMI.')
    heartRate=st.text_input('Number of smoking',help='If you have a blood pressure monitor, use it. If you dont, place your hand on your heart and take 1 minute to count your heartbeats until it reaches 1 minute.')
    glucose=st.text_input('Number of glucose',help='It is in your test.')
    if st.button('Predicting the data'):
        try:
            age = int(age) if age else 0
            male = int(male) if male else 0
            currentSmoker= int(currentSmoker) if currentSmoker else 0
            cigsPerDay = int(cigsPerDay) if cigsPerDay else 0
            BPMeds = int(BPMeds) if BPMeds else 0
            diabetes = int(diabetes) if diabetes else 0
            totChol =int(totChol) if totChol else 0
            sysBP = int(sysBP) if sysBP else 0
            diaBP = int(diaBP) if diaBP else 0
            BMI = int(BMI) if BMI else 0
            heartRate = int(heartRate) if  heartRate else 0
            glucose = int(glucose) if time else 1

            input_data = np.array([[age, male,currentSmoker, cigsPerDay, BPMeds, diabetes, totChol,sysBP, diaBP,BMI,heartRate,glucose]])
            prediction = model.predict(input_data)
            probability = model.predict_proba(input_data)[0][1]

            if prediction[0] == 1:
                st.error(f'Result: Diabetic (Probability: {probability:.2%})')
            else:
                st.success(f'Result: Healthy (Probability: {(1 - probability):.2%})')

        except ValueError:  # اگر کاربر عدد وارد نکند (مثلاً حروف بنویسد)
            st.error('Please enter only numbers.')
        except Exception as e:  # سایر خطاهای غیرمنتظره
            st.error(f'An error has occurred.: {e}')

    st.title('HELP high blood pressure ')
    if st.button('Guide to high blood pressure '):
        st.markdown('''Treatment for high blood pressure involves a combination 
                    of lifestyle changes and, if necessary, medication. Lifestyle 
                    changes include a healthy diet low in salt and fat, regular exercise,
                     weight loss, quitting smoking and alcohol, and stress management. If these
                     changes are not enough, your doctor may prescribe medications to lower your blood pressure.''')
        st.markdown('___Methods of treating high blood pressure___ : ')
        st.markdown('___Lifestyle changes___ : ')
        st.markdown('''Healthy diet: A diet rich in fruits, vegetables, whole grains,
                     and low-fat dairy products and limiting intake of salt, saturated 
                    fats, and cholesterol is essential.''')
        st.markdown('___Regular exercise___ : Regular physical activity such as walking, running, swimming, or cycling helps lower blood pressure.')
        st.markdown('___Weight loss___ : If you are overweight, losing weight can significantly lower blood pressure.')
        st.markdown('___Quit smoking___: Smoking causes an immediate and temporary increase in blood pressure as well as long-term damage to the arteries.')
        st.markdown('___Reduce alcohol consumption___ : Excessive alcohol consumption can raise blood pressure.')
        st.markdown('___Stress management___: Relaxation techniques such as meditation, yoga, and deep breathing can help reduce stress and, consequently, blood pressure.')
        st.markdown('___Medicines___ :')
        st.markdown('''Your doctor may prescribe medications such as ACE inhibitors, 
                    beta blockers, diuretics, or other medications to lower your blood
                     pressure, depending on your individual circumstances. It is essential 
                    to take your medications as directed by your doctor and have regular follow-ups
                     to assess your health.''')
        st.markdown('___Home remedies___ :')
        st.markdown('___Deep breathing___ : Practicing deep breathing can help reduce stress and lower blood pressure.')
        st.markdown('___Increase water intake___ : Drinking enough water helps thin the blood and eliminate sodium from the body.')
        st.markdown('___Avoid stimulants___ : Avoiding caffeine, nicotine, and alcohol can help lower blood pressure.')
        st.markdown('___Consuming beneficial foods___ : Consuming foods like broccoli, bananas, milk, and other foods rich in potassium, magnesium, and calcium can help control blood pressure.')
        st.warning('___Important points___ :')
        st.markdown('''Always consult your doctor before starting any treatment or
                     lifestyle change. Check your blood pressure regularly and see your 
                    doctor if you notice any problems. Treating high blood pressure is a long-term
                     process and requires patience and perseverance.''')
        st.warning('This content is for informational purposes only. For medical advice or diagnosis, please consult a professional. AI responses may contain errors.')