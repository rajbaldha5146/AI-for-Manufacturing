# Smart Dairy Temp Controller 🥛❄️

An AI-powered environment system that recommends optimal room temperatures for storing dairy products based on product requirements and external temperature.

## 📂 Project Structure
- `app.py`: Streamlit app to interact with your AI model.
- `data/`: Contains `dairy_products.csv` (products & their optimal temperature ranges).
- `model/`: Contains the pre-trained model `temp_model.pkl`.
- `requirements.txt`: List of Python dependencies.

## 🚀 Getting Started

1. **Install dependencies**  
   Make sure you have Python installed, then run:
    ```bash
    pip install -r requirements.txt
    ```

2. **Launch the app**  
    ```bash
    streamlit run app.py
    ```

3. **Use the App**
    - Open the link Streamlit provides in your browser.
    - Select the dairy products currently in your storage.
    - Enter the current external temperature.
    - See the recommended room temperature for safe storage!

## 🤖 Features
- Choose one or more dairy products.
- Input external temperature.
- Get AI-powered recommended room temperature instantly.

## 🤝 Contributing
This project is meant for demonstration and practical use. Contributions are welcome to improve features or add integrations like sensors or databases.
