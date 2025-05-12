# Telco Customer Churn Analysis - Streamlit App

This is the Streamlit application for the Telco Customer Churn Analysis project. It provides an interactive interface for exploring customer churn data, viewing analysis results, and making predictions.

## Features

- 📊 Interactive Dashboard
  - Key metrics and visualizations
  - Churn distribution analysis
  - Customer insights
- 🔍 Detailed Data Analysis
  - Customer demographics
  - Service usage patterns
  - Financial analysis
  - Churn patterns
- 🎯 Churn Prediction
  - Single customer prediction
  - Batch prediction via CSV upload
  - Feature importance analysis
- 📈 Performance Metrics
  - Model evaluation metrics
  - Prediction confidence scores
- 💡 Actionable Insights
  - Top churn factors
  - Recommendations
  - Risk assessment

## Setup

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Dashboard**
   - View key metrics and visualizations
   - Explore churn distribution
   - Review customer insights

2. **Data Analysis**
   - Select analysis type from dropdown
   - Explore different aspects of the data
   - View interactive visualizations

3. **Churn Prediction**
   - Single Customer: Enter customer details for individual prediction
   - Batch Prediction: Upload CSV file for multiple predictions
   - Download prediction results

## Data Format

For batch predictions, the CSV file should include the following columns:
- Tenure in Months
- Age
- Internet Type
- Monthly Charge
- Total Revenue
- Payment Method

## Development

The application is built using:
- Streamlit for the web interface
- Plotly for interactive visualizations
- Pandas for data manipulation
- Scikit-learn for machine learning
- Hugging Face for dataset access

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 