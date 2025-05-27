import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.arima.model import ARIMA
import arabic_reshaper
from bidi.algorithm import get_display

# تحميل البيانات
file_path = r"D:\China's Inspiring Experience\GDP World Bank\csv\API_NY.GDP.MKTP.CD_DS2_en_csv_v2_19294.csv"

@st.cache_data
def load_data(path):
    df = pd.read_csv(path, skiprows=4)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # إزالة الأعمدة غير المسماة
    df.fillna(method='ffill', inplace=True)               # ملء القيم الفارغة
    return df

try:
    df = load_data(file_path)
except FileNotFoundError:
    st.error(f"لم يتم العثور على الملف: {file_path}")
    st.stop()

# استخراج بيانات الصين
china = df[df['Country Name'] == 'China'].iloc[:, 4:]
china = china.loc[:, china.columns.str.match(r'^\d+$')]  # الأعمدة الرقمية فقط
china = china.T
china.columns = ['GDP']
china.index.name = 'Year'
china.index = china.index.astype(int)
china.dropna(inplace=True)

# ضبط الخط العربي الصحيح
arabic_font_path = "C:\\Windows\\Fonts\\Arial.ttf"  # يمكنك تجربة Tahoma أو Geeza Pro أيضًا
arabic_font = fm.FontProperties(fname=arabic_font_path)

# إعادة تشكيل النص العربي وتحويله لعرض صحيح
def fix_arabic_text(text):
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)

# واجهة Streamlit
st.markdown("<h1 style='text-align: right; direction: rtl;'>تحليل وتنبؤ الناتج المحلي الإجمالي للصين</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: right; direction: rtl;'>البيانات</h2>", unsafe_allow_html=True)
st.dataframe(china)

# رسم بياني للناتج المحلي الإجمالي
st.markdown("<h2 style='text-align: right; direction: rtl;'>الرسم البياني للناتج المحلي الإجمالي</h2>", unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(china.index, china['GDP'] / 1e12, marker='o', label=fix_arabic_text("الصين"))

# تعيين العناوين باللغة العربية
ax.set_xlabel(fix_arabic_text("السنة"), fontproperties=arabic_font)
ax.set_ylabel(fix_arabic_text("الناتج المحلي (تريليون دولار)"), fontproperties=arabic_font)
ax.set_title(fix_arabic_text("الناتج المحلي الإجمالي للصين"), fontproperties=arabic_font)
ax.legend(prop=arabic_font)
ax.grid(True)

st.pyplot(fig)

# توقع باستخدام Polynomial Regression


st.markdown(f"<h2 style='text-align: right; direction: rtl;'>توقع الناتج المحلي - Polynomial Regression</h2>",unsafe_allow_html=True)

years = china.index.values.reshape(-1, 1)
gdp_values = china['GDP'].values

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(years)
model = LinearRegression()
model.fit(X_poly, gdp_values)

future_years = np.arange(china.index[-1] + 1, china.index[-1] + 6).reshape(-1, 1)
future_X_poly = poly.transform(future_years)
future_preds = model.predict(future_X_poly)

future_df = pd.DataFrame({'Year': future_years.flatten(), 'Predicted GDP': future_preds})
st.write(future_df)

# رسم بياني لتوقعات Polynomial Regression
st.markdown("<h2 style='text-align: right; direction: rtl;'>توقع الناتج المحلي - Polynomial Regression</h2>", unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(china.index, china['GDP'] / 1e12, marker='o', label=fix_arabic_text("الصين"))
ax.plot(future_years.flatten(), future_preds / 1e12, marker='x', linestyle="dashed", label=fix_arabic_text("التوقع"))

ax.set_xlabel(fix_arabic_text("السنة"), fontproperties=arabic_font)
ax.set_ylabel(fix_arabic_text("الناتج المحلي (تريليون دولار)"), fontproperties=arabic_font)
ax.set_title(fix_arabic_text("توقعات Polynomial Regression"), fontproperties=arabic_font)
ax.legend(prop=arabic_font)
ax.grid(True)

st.pyplot(fig)

# توقع باستخدام ARIMA

st.markdown("<h2 style='text-align: right; direction: rtl;'>توقع الناتج المحلي - ARIMA</h2>",unsafe_allow_html=True)
model_arima = ARIMA(china['GDP'], order=(1, 1, 1))
model_fit_arima = model_arima.fit()
forecast_arima = model_fit_arima.forecast(steps=5)

arima_df = pd.DataFrame({'Year': range(china.index[-1] + 1, china.index[-1] + 6),'Predicted GDP': forecast_arima.values})
st.write(arima_df)

# رسم بياني لتوقعات ARIMA

st.markdown("<h2 style='text-align: right; direction: rtl;'>الرسم البياني لتوقعات ARIMA</h2>",unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(china.index, china['GDP'] / 1e12, marker='o', label=fix_arabic_text("الصين"))
ax.plot(arima_df['Year'], arima_df['Predicted GDP'] / 1e12, marker='s', linestyle="dashed", label=fix_arabic_text("التوقع"))

ax.set_xlabel(fix_arabic_text("السنة"), fontproperties=arabic_font)
ax.set_ylabel(fix_arabic_text("الناتج المحلي (تريليون دولار)"), fontproperties=arabic_font)
ax.set_title(fix_arabic_text("توقعات ARIMA"), fontproperties=arabic_font)
ax.legend(prop=arabic_font)
ax.grid(True)

st.pyplot(fig)

# تشغيل Streamlit وعرض البيانات والرسوم البيانية
st.pyplot(fig)

import time
from selenium import webdriver

driver = webdriver.Chrome()
driver.get("http://localhost:8501")  # ضع رابط تطبيق Streamlit هنا

time.sleep(5)  # انتظار تحميل الصفحة بالكامل
driver.save_screenshot("streamlit_page.png")
driver.quit()