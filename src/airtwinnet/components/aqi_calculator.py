class AQICalculator:

    def calculate_pm25_aqi(self, pm25):

        if pm25 <= 12.0:
            return pm25 * 50 / 12.0

        elif pm25 <= 35.4:
            return 50 + (pm25 - 12.1) * 50 / (35.4 - 12.1)

        elif pm25 <= 55.4:
            return 100 + (pm25 - 35.5) * 50 / (55.4 - 35.5)

        elif pm25 <= 150.4:
            return 150 + (pm25 - 55.5) * 50 / (150.4 - 55.5)

        elif pm25 <= 250.4:
            return 200 + (pm25 - 150.5) * 50 / (250.4 - 150.5)

        else:
            return 300


    def calculate_pm10_aqi(self, pm10):

        if pm10 <= 54:
            return pm10 * 50 / 54

        elif pm10 <= 154:
            return 50 + (pm10 - 55) * 50 / (154 - 55)

        elif pm10 <= 254:
            return 100 + (pm10 - 155) * 50 / (254 - 155)

        elif pm10 <= 354:
            return 150 + (pm10 - 255) * 50 / (354 - 255)

        elif pm10 <= 424:
            return 200 + (pm10 - 355) * 50 / (424 - 355)

        else:
            return 300


    def calculate_aqi(self, pm25, pm10):

        pm25_aqi = self.calculate_pm25_aqi(pm25)
        pm10_aqi = self.calculate_pm10_aqi(pm10)

        aqi = max(pm25_aqi, pm10_aqi)

        return round(aqi, 2)


    def get_category(self, aqi):

        if aqi <= 50:
            return "Good"

        elif aqi <= 100:
            return "Satisfactory"

        elif aqi <= 200:
            return "Moderate"

        elif aqi <= 300:
            return "Poor"

        elif aqi <= 400:
            return "Very Poor"

        else:
            return "Severe"


if __name__ == "__main__":

    calculator = AQICalculator()

    pm25 = 34.264
    pm10 = 56.328

    aqi = calculator.calculate_aqi(
        pm25,
        pm10
    )

    category = calculator.get_category(aqi)

    print("Predicted PM2.5:", pm25)
    print("Predicted PM10:", pm10)
    print("Calculated AQI:", aqi)
    print("AQI Category:", category)