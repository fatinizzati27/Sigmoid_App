#Sigmoid_App
 
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sigmoid Activation Function")

st.title("Sigmoid Activation Function")
st.write("Sigmoid maps any input to a value between 0 and 1, making it useful for probability-based predictions.")

x = np.linspace(-10, 10, 400)
y = 1 / (1 + np.exp(-x))

fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_xlabel("Weighted Sum (z)")
ax.set_ylabel("Activation Output")
ax.set_title("Sigmoid: f(z) = 1 / (1 + e^-z)")

st.pyplot(fig)
