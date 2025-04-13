import streamlit as st
import joblib
import pandas as pd

# Load all encoders and model
body_onec = joblib.load('body_onec.pkl')
diet_onec = joblib.load('diet_onec.pkl')
shower_onec = joblib.load('shower_onec.pkl')
heating_onec = joblib.load('heating_onec.pkl')
transport_onec = joblib.load('transport_onec.pkl')
vehicle_onec = joblib.load('vehicle_onec.pkl')
social_onec = joblib.load('social_onec.pkl')
freq_onec = joblib.load('freq_onec.pkl')
wasteSize_onec = joblib.load('wasteSize_onec.pkl')
energyEff_onec = joblib.load('energyEff_onec.pkl')
Sex_encoder = joblib.load('Sex_encoder.pkl')
recycle_bin = joblib.load('recycle_bin.pkl')
cook_bin = joblib.load('cook_bin.pkl')
model = joblib.load('your_model.pkl')  # Replace with your model file

def validate_inputs(inputs):
    # Transport validation
    if inputs['Transport'] == 'walk/bicycle' and inputs['Vehicle Monthly Distance Km'] > 0:
        st.error("❌ Vehicle distance must be 0 when transport is walk/bicycle")
        return False
        
    if inputs['Transport'] != 'private' and inputs['Vehicle Type'] != 'Unknown':
        st.error("❌ Vehicle type should be 'Unknown' for non-private transport")
        return False
        
    # Sanity checks
    if inputs['How Long TV PC Daily Hour'] + inputs['How Long Internet Daily Hour'] > 24:
        st.error("❌ Total screen time cannot exceed 24 hours per day")
        return False
        
    if inputs['Monthly Grocery Bill'] < 50:
        st.warning("⚠️ Very low grocery bill - double check amount")
        
    if inputs['Body Type'] == 'obese' and inputs['How Often Shower'] == 'twice a day':
        st.warning("⚠️ Frequent showers with obese body type might affect accuracy")
        
    return True

def main():
    st.title("🌍 Carbon Emission Calculator")
    st.markdown("### Estimate your personal carbon footprint based on lifestyle choices")
    
    with st.form("input_form"):
        st.header("Personal Information")
        col1, col2 = st.columns(2)
        
        with col1:
            sex = st.selectbox("Gender", ['male', 'female'], help="Biological sex for metabolic calculations")
            body_type = st.selectbox("Body Type", ['normal', 'obese', 'overweight', 'underweight'])
            
        with col2:
            diet = st.selectbox("Diet", ['omnivore', 'pescatarian', 'vegan', 'vegetarian'])
            shower_freq = st.selectbox("Shower Frequency", ['daily', 'less frequently', 'more frequently', 'twice a day'])
        
        st.header("Living Situation")
        col3, col4 = st.columns(2)
        
        with col3:
            heating_source = st.selectbox("Home Heating Source", ['coal', 'electricity', 'natural gas', 'wood'])
            transport = st.selectbox("Primary Transportation", ['private', 'public', 'walk/bicycle'])
            
        with col4:
            vehicle_type = 'Unknown'
            if transport == 'private':
                vehicle_type = st.selectbox("Vehicle Fuel Type", ['diesel', 'electric', 'hybrid', 'lpg', 'petrol'])
                
            waste_size = st.selectbox("Typical Waste Bag Size", ['extra large', 'large', 'medium', 'small'])
        
        st.header("Lifestyle Habits")
        col5, col6 = st.columns(2)
        
        with col5:
            social_activity = st.selectbox("Social Activity Frequency", ['never', 'often', 'sometimes'])
            air_travel = st.selectbox("Air Travel Frequency", ['frequently', 'never', 'rarely', 'very frequently'])
            
        with col6:
            energy_efficiency = st.selectbox("Energy Efficient Practices", ['No', 'Sometimes', 'Yes'])
            waste_count = st.number_input("Weekly Waste Bags", min_value=0, max_value=20, value=2)
        
        st.header("Consumption Patterns")
        col7, col8 = st.columns(2)
        
        with col7:
            grocery_bill = st.number_input("Monthly Grocery Bill ($)", min_value=0, max_value=10000, value=300)
            vehicle_distance = st.number_input(
                "Monthly Vehicle Distance (km)", 
                min_value=0, 
                max_value=20000,
                value=0,
                disabled=(transport != 'private')
            )
            
        with col8:
            tv_hours = st.number_input("Daily Screen Time (TV/PC)", min_value=0, max_value=24, value=4)
            internet_hours = st.number_input("Daily Internet Use (hours)", min_value=0, max_value=24, value=3)
        
        new_clothes = st.number_input("New Clothing Items/Month", min_value=0, max_value=100, value=5)
        
        st.header("Sustainability Practices")
        col9, col10 = st.columns(2)
        
        with col9:
            recycling = st.multiselect(
                "Recycled Materials", 
                ['Glass', 'Metal', 'Paper', 'Plastic'],
                default=['Paper'],
                max_selections=4
            )
            
        with col10:
            cooking = st.multiselect(
                "Cooking Appliances Used", 
                ['Airfryer', 'Grill', 'Microwave', 'Oven', 'Stove'],
                default=['Stove'],
                max_selections=5
            )
        
        submitted = st.form_submit_button("Calculate My Carbon Footprint")
    
    if submitted:
        input_data = {
            'Sex': sex,
            'Monthly Grocery Bill': grocery_bill,
            'Vehicle Monthly Distance Km': vehicle_distance,
            'Waste Bag Weekly Count': waste_count,
            'How Long TV PC Daily Hour': tv_hours,
            'How Many New Clothes Monthly': new_clothes,
            'How Long Internet Daily Hour': internet_hours,
            'Body Type': body_type,
            'Diet': diet,
            'How Often Shower': shower_freq,
            'Heating Energy Source': heating_source,
            'Transport': transport,
            'Vehicle Type': vehicle_type,
            'Social Activity': social_activity,
            'Frequency of Traveling by Air': air_travel,
            'Waste Bag Size': waste_size,
            'Energy efficiency': energy_efficiency,
            'Recycling': recycling,
            'Cooking_With': cooking
        }
        
        if not validate_inputs(input_data):
            return
            
        with st.spinner("Calculating your carbon footprint..."):
            try:
                # Create DataFrame
                df = pd.DataFrame([input_data])
                
                # Encode categorical features
                df['Sex'] = Sex_encoder.transform(df['Sex'])
                
                # One-hot encoding
                body_encoded = body_onec.transform(df[['Body Type']]).toarray()
                diet_encoded = diet_onec.transform(df[['Diet']]).toarray()
                shower_encoded = shower_onec.transform(df[['How Often Shower']]).toarray()
                heating_encoded = heating_onec.transform(df[['Heating Energy Source']]).toarray()
                transport_encoded = transport_onec.transform(df[['Transport']]).toarray()
                vehicle_encoded = vehicle_onec.transform(df[['Vehicle Type']]).toarray()
                social_encoded = social_onec.transform(df[['Social Activity']]).toarray()
                freq_encoded = freq_onec.transform(df[['Frequency of Traveling by Air']]).toarray()
                waste_encoded = wasteSize_onec.transform(df[['Waste Bag Size']]).toarray()
                energy_encoded = energyEff_onec.transform(df[['Energy efficiency']]).toarray()
                
                # Multi-label encoding
                df['Recycling'] = df['Recycling'].apply(lambda x: x if isinstance(x, list) else [])
                df['Cooking_With'] = df['Cooking_With'].apply(lambda x: x if isinstance(x, list) else [])
                
                recycle_encoded = recycle_bin.transform(df['Recycling'])
                cook_encoded = cook_bin.transform(df['Cooking_With'])
                
                # Combine features
                numerical_features = df[['Sex', 'Monthly Grocery Bill', 'Vehicle Monthly Distance Km',
                                        'Waste Bag Weekly Count', 'How Long TV PC Daily Hour',
                                        'How Many New Clothes Monthly', 'How Long Internet Daily Hour']]
                
                final_df = pd.concat([
                    numerical_features.reset_index(drop=True),
                    pd.DataFrame(body_encoded, columns=body_onec.get_feature_names_out()),
                    pd.DataFrame(diet_encoded, columns=diet_onec.get_feature_names_out()),
                    pd.DataFrame(shower_encoded, columns=shower_onec.get_feature_names_out()),
                    pd.DataFrame(heating_encoded, columns=heating_onec.get_feature_names_out()),
                    pd.DataFrame(transport_encoded, columns=transport_onec.get_feature_names_out()),
                    pd.DataFrame(vehicle_encoded, columns=vehicle_onec.get_feature_names_out()),
                    pd.DataFrame(social_encoded, columns=social_onec.get_feature_names_out()),
                    pd.DataFrame(freq_encoded, columns=freq_onec.get_feature_names_out()),
                    pd.DataFrame(waste_encoded, columns=wasteSize_onec.get_feature_names_out()),
                    pd.DataFrame(energy_encoded, columns=energyEff_onec.get_feature_names_out()),
                    pd.DataFrame(recycle_encoded, columns=recycle_bin.classes_),
                    pd.DataFrame(cook_encoded, columns=cook_bin.classes_)
                ], axis=1)
                
                # Ensure column order matches training
                # final_df = final_df.reindex(columns=model.feature_names, axis=1)
                
                # Make prediction
                prediction = model.predict(final_df)
                
                # Display results
                st.success(f"Your Estimated Carbon Footprint: **{prediction[0]:,.2f} kg CO₂/year**")
                
                # Interpretation
                st.markdown("""
                **Footprint Classification:**
                - 🌱 **Low (< 1,000 kg):** Below average environmental impact
                - 🌍 **Average (1,000-3,000 kg):** Typical urban dweller
                - 🔥 **High (> 3,000 kg):** Above average impact - consider reduction strategies
                """)
                
                st.markdown("---")
                st.info("💡 **Tips to Reduce Your Footprint:**\n"
                        "- Reduce meat consumption\n"
                        "- Use public transportation more frequently\n"
                        "- Improve home insulation\n"
                        "- Switch to renewable energy sources")
                        
            except Exception as e:
                st.error(f"Error in calculation: {str(e)}")
                st.stop()

if __name__ == "__main__":
    main()