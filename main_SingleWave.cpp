#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_ADDR 0x3C

// Pin configuration
const int analogPin      = 3;   // ADC pin for voltage measurement
const int emitterPin     = 45;  // PWM-controlled emitter
const int statusLedPin   = 41;  // Status LED
const int sdaPin         = 14;  // I2C SDA
const int sclPin         = 15;  // I2C SCL

// PWM settings
const int EMITTER_OFF = 50;
const int EMITTER_ON  = 250;

// ADC & filtering
#define MA_WINDOW_SIZE 5    // Moving Average window size
#define TOTAL_SAMPLES   9   // Samples for median filter (must be odd)
#define AVG_SAMPLES    10   // Glucose readings to average per measurement

// Display & timing
const unsigned long DISPLAY_INTERVAL_MS = 10000UL;  // measurement interval
unsigned long lastMeasurementTime = 0;

// Glucose normal range (mg/dL)
const float NORMAL_LOW  = 70.0;
const float NORMAL_HIGH = 140.0;

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire);

// LED blink timing
unsigned long previousBlink = 0;
const int blinkInterval = 250;  // ms

// Last measured glucose
float lastAvgGlucose = 0;

// Compute moving average of ADC
float movingAverageADC() {
  long sum = 0;
  for (int i = 0; i < MA_WINDOW_SIZE; i++) {
    sum += analogRead(analogPin);
    delayMicroseconds(100);
  }
  return float(sum) / MA_WINDOW_SIZE;
}

// Median-filtered voltage
float medianFilteredVoltage() {
  float samples[TOTAL_SAMPLES];
  for (int i = 0; i < TOTAL_SAMPLES; i++) {
    samples[i] = movingAverageADC() * (3.3 / 4095.0);
  }
  for (int i = 0; i < TOTAL_SAMPLES - 1; i++) {
    for (int j = i + 1; j < TOTAL_SAMPLES; j++) {
      if (samples[j] < samples[i]) {
        float tmp = samples[i]; samples[i] = samples[j]; samples[j] = tmp;
      }
    }
  }
  return samples[TOTAL_SAMPLES / 2];
}

// UI: display glucose, status, and time since last update
void updateDisplay(float glucose, const char* status, unsigned int secsSince) {
  display.clearDisplay();
  display.setTextSize(1);  // uniform size for all text

  // Glucose reading with label
  display.setCursor(0, 0);
  display.print("Glucose: ");
  display.print(glucose, 1);
  display.println(" mg/dL");

  // Status line
  display.setCursor(0, 16);
  display.print("Status: ");
  display.println(status);

  // Time since last update
  display.setCursor(0, 32);
  display.print("Updated: ");
  display.print(secsSince);
  display.println(" s ago");

  display.display();
}

// Blink the status LED
void blinkStatusLed() {
  if (millis() - previousBlink >= blinkInterval) {
    digitalWrite(statusLedPin, !digitalRead(statusLedPin));
    previousBlink = millis();
  }
}

void setup() {
  // I2C & OLED
  Wire.setPins(sdaPin, sclPin);
  Wire.begin();
  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    pinMode(statusLedPin, OUTPUT);
    while (1) {
      digitalWrite(statusLedPin, !digitalRead(statusLedPin));
      delay(100);
    }
  }

  pinMode(emitterPin, OUTPUT);
  pinMode(statusLedPin, OUTPUT);

  analogReference(AR_VDD4);
  analogReadResolution(12);

  display.setTextColor(SSD1306_WHITE);
  display.clearDisplay();
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.println("Initializing...");
  display.display();
  delay(1000);
}

void loop() {
  blinkStatusLed();
  unsigned long now = millis();

  // Time to take a new measurement?
  if (now - lastMeasurementTime >= DISPLAY_INTERVAL_MS) {
    lastMeasurementTime = now;
    float sumGlucose = 0;
    for (int i = 0; i < AVG_SAMPLES; i++) {
      analogWrite(emitterPin, EMITTER_OFF);
      delay(100);
      float voltsOff = medianFilteredVoltage();

      analogWrite(emitterPin, EMITTER_ON);
      delay(100);
      float voltsOn = medianFilteredVoltage();

      float diff_mV   = (voltsOn - voltsOff) * 1000.0;
      float glucose_i = -0.088 * diff_mV + 176.2; //model
      sumGlucose += glucose_i;
    }
    lastAvgGlucose = sumGlucose / AVG_SAMPLES;
  }

  // Determine status
  const char* status;
  if (lastAvgGlucose < NORMAL_LOW)         status = "LOW";
  else if (lastAvgGlucose > NORMAL_HIGH)   status = "HIGH";
  else                                      status = "NORMAL";

  unsigned int secsSince = (millis() - lastMeasurementTime) / 1000;
  updateDisplay(lastAvgGlucose, status, secsSince);

  delay(100);  // loop pacing
}
