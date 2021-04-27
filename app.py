import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as px
import joblib
from db_connect import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib
import math
matplotlib.use('Agg')


jenis_dinding = {
    '1. Tembok': 1,
    '2. Plesteran anyaman bambu/kawat': 2,
    '3. Kayu/papan': 3,
    '4. Anyaman bambu': 4,
    '5. Batang kayu': 5,
    '6. Bambu': 6,
    '7. Lainnya': 7
}

jenis_lantai = {
    '1. Marmer/granit': 1,
    '2. Keramik': 2,
    '3. Parket/vinil/karpet': 3,
    '4. Ubin/tegel/teraso': 4,
    '5. Kayu/papan': 5,
    '6. Semen/bata merah': 6,
    '7. Bambu': 7,
    '8. Tanah': 8,
    '9. Lainnya': 9
}

kepemilikan_ftbab = {
    '1. Ada, digunakan hanya ART sendiri': 1,
    '2. Ada, digunakan bersama ART rumah tangga tertentu': 2,
    '3. Ada, di MCK komunal': 3,
    '4. Ada, di MCK umum/siapapun menggunakan': 4,
    '5. Ada, ART tidak menggunakan': 5,
    '6. Tidak ada fasilitas': 6
}

sumber_air_minum = {
    '1. Air kemasan bermerk': 1,
    '2. Air isi ulang': 2,
    '3. Leding': 3,
    '4. Sumur bor/pompa': 4,
    '5. Sumur terlindung': 5,
    '6. Sumur tak terlindung': 6,
    '7. Mata air terlindung': 7,
    '8. Mata air tak terlindung': 8,
    '9. Air permukaan (sungai/danau/waduk/kolam/irigasi)': 9,
    '10. Air hujan': 10,
    '11. Lainnya': 11
}

sumber_penerangan = {
    '1. Listrik PLN dengan meteran': 1,
    '2. Listrik PLN tanpa meteran': 2,
    '3. Listrik non PLN': 3,
    '4. Bukan listrik': 4
}

bahan_bakar = {
    '1. Listrik': 1,
    '2. Elpiji 5,5 kg/blue gas': 2,
    '3. Elpiji 12 kg': 3,
    '4. Elpiji 3 kg': 4,
    '5. Gas kota': 5,
    '6. Biogas': 6,
    '7. Minyak tanah': 7,
    '8. Briket': 8,
    '9. Arang': 9,
    '10. Kayu bakar': 10,
    '11. Lainnya': 11,
    '0. Tidak memasak di rumah': 0
}


def get_value(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value


def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return key


def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), 'rb'))
    return loaded_model


def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)

    menu = ['Home', 'Login', 'Sign Up']
    submenu = ['Plot', 'Prediction']

    choice = st.sidebar.selectbox("Menu", menu)
    if choice == 'Home':
        st.subheader('Selamat datang di')
        st.title("'Poor' Classification App ")

        st.write("'Poor' Classification App adalah aplikasi berbasis web untuk mengklasifikasikan status penduduk apakah tergolong Rumah Tangga 'Miskin' atau 'Tidak Miskin'  di Sulawesi Tenggara Ditinjau dari Kondisi Perumahan menggunakan Mechine Learning Model.")

        st.write('Sumber data (dataset) yang digunakan dalam aplikasi ini berasal dari SUSENAS yang dilakukan oleh Badan Pusat Statistik Kabupaten Wakatobi 2019.')

        st.write('SUSENAS sendiri merupakan survei yang dirancang untuk mengumpulkan data sosial kependudukan yang relatif sangat luas. Data yang dikumpulkan antara lain menyangkut bidang-bidang pendidikan, kesehatan/gizi, perumahan, sosial ekonomi lainnya, kegiatan sosial budaya, konsumsi/pengeluaran dan pendapatan rumah tangga, perjalanan, dan pendapat masyarakat mengenai kesejahteraan rumah tangganya (https://www.bps.go.id/index.php/subjek/81).')

    elif choice == 'Login':
        st.title("'Poor' Classification App ")

        username = st.sidebar.text_input('Username')
        password = st.sidebar.text_input('Password', type='password')
        st.sidebar.info('Sign Up first if you do not have an account')
        if st.sidebar.checkbox('Login'):
            result = login_user(username, password)
            if result:
                st.success(f'Welcome {username}')

                activity = st.selectbox('Activity', submenu)
                if activity == 'Plot':
                    st.subheader('Data Visualisation Plot')
                    df = pd.read_csv('data/clean_bps2019data.csv')
                    st.dataframe(df)

                    features = ['Luas lantai rumah bangunan tempat tinggal (m2)', 'Jenis dinding rumah paling dominan/luas', 'Jenis lantai paling dominan/luas', 'Kepemilikan Fasilitas BAB dan siapa saja yang menggunakan',
                                'Sumber air minum', 'Sumber utama penenrangan rumah tangga', 'Bahan bakar utama untuk memasak', 'Banyak Anggota Rumah Tangga', 'Miskin(0) atau tidak miskin(1)']
                    jumlah_data = df.shape[0]
                    jumlah_features = df.shape[1]
                    st.subheader('Keterangan')
                    ket = {
                        'Jumlah data': jumlah_data,
                        'Jumlah features': jumlah_features,
                        'Keterangan features': {
                            df.columns[0]: features[0],
                            df.columns[1]: features[1],
                            df.columns[2]: features[2],
                            df.columns[3]: features[3],
                            df.columns[4]: features[4],
                            df.columns[5]: features[5],
                            df.columns[6]: features[6],
                            df.columns[7]: features[7],
                            df.columns[8]: features[8],
                        }

                    }

                    st.json(ket)

                    st.subheader('Keterangan Statistik data')
                    st.write(df.describe())

                    data = pd.read_csv('data/data_convert_bps2019.csv')

                    st.line_chart(data['Luas lantai (m2)'].value_counts())

                    st.bar_chart(data['jenis dinding'].value_counts())
                    st.bar_chart(data['jenis lantai'].value_counts())
                    st.bar_chart(data['kepemilikan ftbab'].value_counts())
                    st.bar_chart(data['sumber ir minum'].value_counts())
                    st.bar_chart(data['sumber penerangan'].value_counts())
                    st.bar_chart(data['bahan bakar'].value_counts())
                    st.line_chart(data['Banyak ART'].value_counts())
                    st.bar_chart(data['outcome'].value_counts())

                    names = ['Tidak Miskin', 'Miskin']
                    fig = px.pie(df, values=data['outcome'].value_counts(
                    ), names=names, title='Presentasi Penduduk Miskin dan Tidak Miskin')
                    st.plotly_chart(fig)

                elif activity == 'Prediction':
                    st.subheader('Classification')

                    luas_lantai = st.number_input(
                        'Luas Lantai Rumah Bangunan Tempat Tinggal (m2)', 5, 516)
                    j_dinding = st.selectbox(
                        'Jenis Dinding Rumah Paling Dominan/Luas', tuple(jenis_dinding.keys()))
                    j_lantai = st.selectbox(
                        'Jenis Lantai Paling Dominan/Luas', tuple(jenis_lantai.keys()))
                    k_ftbab = st.selectbox('Kepemilikan Fasilitas BAB dan Siapa saja yang menggunakan', tuple(
                        kepemilikan_ftbab.keys()))
                    air_minum = st.selectbox(
                        'Sumber Air Minum', tuple(sumber_air_minum.keys()))
                    penerangan = st.selectbox(
                        'Sumber Utama Penerangan Rumah Tangga', tuple(sumber_penerangan.keys()))
                    b_bakar = st.selectbox(
                        'Bahan Bakar Utama Untuk Memasak', tuple(bahan_bakar.keys()))
                    banyak_art = st.slider(
                        'Banyaknya Anggota Rumah Tangga (ART)', 1, 16)

                    feature_list = [luas_lantai, get_value(j_dinding, jenis_dinding), get_value(j_lantai, jenis_lantai), get_value(k_ftbab, kepemilikan_ftbab), get_value(
                        air_minum, sumber_air_minum), get_value(penerangan, sumber_penerangan), get_value(b_bakar, bahan_bakar), banyak_art]

                    # st.write(feature_list)
                    st.write("User input: ")
                    pretty_result = {
                        'Luas Lantai Rumah Bangunan Tempat Tinggal (m2)': luas_lantai,
                        'Jenis Dinding Rumah Paling Dominan/Luas': j_dinding,
                        'Jenis Lantai Paling Dominan/Luas': j_lantai,
                        'Kepemilikan Fasilitas BAB dan Siapa saja yang menggunakan': k_ftbab,
                        'Sumber Air Minum': air_minum,
                        'Sumber Utama Penerangan Rumah Tangga': penerangan,
                        'Bahan Bakar Utama Untuk Memasak': b_bakar,
                        'Banyaknya Anggota Rumah Tangga (ART)': banyak_art
                    }
                    st.json(pretty_result)
                    data = {
                        'Luas lantai (m2)': feature_list[0],
                        'Jenis dinding': feature_list[1],
                        'Jenis Lantai': feature_list[2],
                        'Kepemilikan FTBAB': feature_list[3],
                        'Sumber air minum': feature_list[4],
                        'Sumber penerangan': feature_list[5],
                        'Bahan bakar': feature_list[6],
                        'Banyak ART': feature_list[7]
                    }

                    user_input = pd.DataFrame(data, index=[0])
                    st.dataframe(user_input)
                    single_input = np.array(user_input).reshape(1, -1)

    # ========================================================================================
                    # if st.button('Predict'):
                    #     loaded_model = ''
                    #     prediction = ''
                    #     pred_prob = ''
                    #     if models == 'RandomForest':
                    #         loaded_model = load_model('models/rf2_model')
                    #         prediction = loaded_model.predict(single_input)
                    #         pred_prob = loaded_model.predict_proba(
                    #             single_input)
                    #     elif models == 'SVM':
                    #         loaded_model = load_model('models/svm_model.pkl')
                    #         prediction = loaded_model.predict(single_input)
                    #         pred_prob = loaded_model.predict_proba(
                    #             single_input)
                    #     elif models == 'KNN':
                    #         loaded_model = load_model('models/knn_model.pkl')
                    #         prediction = loaded_model.predict(single_input)
                    #         pred_prob = loaded_model.predict_proba(
                    #             single_input)

                    #     st.subheader('Prediction result:')
                    #     st.write(prediction)

                    #     if prediction == 1:
                    #         st.success('Tidak Miskin')
                    #     else:
                    #         st.warning('Kategori Miskin')

                    #     pred_probab_score = {
                    #         'Tidak Miskin': pred_prob[0][1],
                    #         'Miskin': pred_prob[0][0]*100}

                    #     st.subheader(f'Prediction probability Score {models}')
                    #     st.json(pred_probab_score)
  # ========================================================================================
                    if st.button('Classify'):
                        # loaded_model = load_model('models/rf2_model.pkl')
                        # prediction = loaded_model.predict(single_input)
                        # pred_prob = loaded_model.predict_proba(single_input)
                        # st.subheader('Prediction result:')
                        # st.write(prediction)
                        data = pd.read_csv('data/clean_bps2019data.csv')
                        X = data.iloc[:, 0:8].values
                        y = data.iloc[:, -1].values

                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=1)

                        model = RandomForestClassifier()
                        model.fit(X_train, y_train)

                        X_test_pred = model.predict(X_test)
                        X_test_acc = accuracy_score(X_test_pred, y_test)

                        prediction = model.predict(single_input)
                        pred_prob = model.predict_proba(single_input)
                        st.write(prediction)
                        if prediction == 1:
                            st.success('Tidak Miskin')
                        else:
                            st.warning('Kategori Miskin')

                        pred_probab_score = {
                            'Tidak Miskin': pred_prob[0][1]*100, 'Miskin': pred_prob[0][0]*100}

                        st.subheader('Classifiation probability Score')
                        st.json(pred_probab_score)

                        st.subheader('Model Accuracy')
                        st.write(f"{X_test_acc*100} %")

            else:
                st.warning('Incorrect Password or Username')

    elif choice == 'Sign Up':
        st.title("'Poor' Classification App ")

        st.write("'Poor' Classification App adalah aplikasi berbasis web untuk mengklasifikasikan status penduduk apakah tergolong Rumah Tangga 'Miskin' atau 'Tidak Miskin'  di Sulawesi Tenggara Ditinjau dari Kondisi Perumahan menggunakan Mechine Learning Model.")

        st.write('Sumber data (dataset) yang digunakan dalam aplikasi ini berasal dari SUSENAS yang dilakukan oleh Badan Pusat Statistik Kabupaten Wakatobi 2019.')

        st.write('SUSENAS sendiri merupakan survei yang dirancang untuk mengumpulkan data sosial kependudukan yang relatif sangat luas. Data yang dikumpulkan antara lain menyangkut bidang-bidang pendidikan, kesehatan/gizi, perumahan, sosial ekonomi lainnya, kegiatan sosial budaya, konsumsi/pengeluaran dan pendapatan rumah tangga, perjalanan, dan pendapat masyarakat mengenai kesejahteraan rumah tangganya (https://www.bps.go.id/index.php/subjek/81).')

        new_username = st.sidebar.text_input('Username')
        new_pass = st.sidebar.text_input('Password', type='password')
        confirm_pass = st.sidebar.text_input(
            'Confirm Password', type='password')

        if new_pass == confirm_pass:
            st.sidebar.success('Password Confirmed')

            if st.sidebar.button('Submit'):
                adduser(new_username, new_pass)
                st.success(
                    "Succes! Login to Get Started")

        else:
            st.sidebar.warning('Password not the same')


if __name__ == '__main__':
    main()
