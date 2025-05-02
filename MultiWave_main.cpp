#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_ADDR 0x3C

// I2C Pins
const int sdaPin = 14;
const int sclPin = 15;

// Emitters
const int emitterPins[3] = {45, 43, 47};  // Emitter GPIOs
const int EMITTER_OFF    = 50;
const int EMITTER_ON     = 250;

// ADC Pins
const int analogPins[3] = {3, 29, 30};     // AIN1, AIN5, AIN6

// Status LED
const int statusLedPin = 41;

// Display and timing
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire);
const unsigned long DISPLAY_INTERVAL_MS = 10000UL; // 10s interval
unsigned long lastMeasurementTime = 0;

// Glucose normal range
const float NORMAL_LOW  = 70.0;
const float NORMAL_HIGH = 140.0;

// Filtering parameters
#define MA_WINDOW_SIZE 5
#define TOTAL_SAMPLES   9
#define AVG_SAMPLES    10  // number of model samples (per window)

// Timing for LED
unsigned long previousBlink = 0;
const int blinkInterval = 250;

// Last computed glucose
float lastGlucose = 0;

// Compute moving average on specified pin
float movingAverageADC(int pin) {
  long sum = 0;
  for (int i = 0; i < MA_WINDOW_SIZE; i++) {
    sum += analogRead(pin);
    delayMicroseconds(100);
  }
  return float(sum) / MA_WINDOW_SIZE;
}

// Median-filtered voltage on pin
float medianFilteredVoltage(int pin) {
  float samples[TOTAL_SAMPLES];
  for (int i = 0; i < TOTAL_SAMPLES; i++) {
    samples[i] = movingAverageADC(pin) * (3.3 / 4095.0);
  }
  // Sort
  for (int i = 0; i < TOTAL_SAMPLES - 1; i++) {
    for (int j = i + 1; j < TOTAL_SAMPLES; j++) {
      if (samples[j] < samples[i]) {
        float tmp = samples[i]; samples[i] = samples[j]; samples[j] = tmp;
      }
    }
  }
  return samples[TOTAL_SAMPLES / 2];
}

// Blink status LED
void blinkStatusLed() {
  if (millis() - previousBlink >= blinkInterval) {
    digitalWrite(statusLedPin, !digitalRead(statusLedPin));
    previousBlink = millis();
  }
}

// Update display with glucose, status, and elapsed time
void updateDisplay(float glucose, const char* status, unsigned int secsSince) {
  display.clearDisplay();
  display.setTextSize(1);  
  display.setCursor(0, 0);
  display.print("Glucose: ");
  display.print(glucose, 1);
  display.println(" mg/dL");

  display.setCursor(0, 16);
  display.print("Status: ");
  display.println(status);

  display.setCursor(0, 32);
  display.print("Updated: ");
  display.print(secsSince);
  display.println(" s ago");

  display.display();
}

void setup() {
  // I2C for OLED
  Wire.setPins(sdaPin, sclPin);
  Wire.begin();
  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    pinMode(statusLedPin, OUTPUT);
    while (1) {
      digitalWrite(statusLedPin, !digitalRead(statusLedPin));
      delay(100);
    }
  }
  display.setTextColor(SSD1306_WHITE);

  // Configure pins
  pinMode(statusLedPin, OUTPUT);
  analogReference(AR_VDD4);
  analogReadResolution(12);
  for (int i = 0; i < 3; i++) {
    pinMode(emitterPins[i], OUTPUT);
  }

  // Initial message
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

  if (now - lastMeasurementTime >= DISPLAY_INTERVAL_MS) {
    lastMeasurementTime = now;
    float sumModel = 0;
    // Average multiple model outputs
    for (int s = 0; s < AVG_SAMPLES; s++) {
      float diffs_mV[3];
      for (int ch = 0; ch < 3; ch++) {
        analogWrite(emitterPins[ch], EMITTER_OFF);
        delay(50);
        float vo = medianFilteredVoltage(analogPins[ch]);
        analogWrite(emitterPins[ch], EMITTER_ON);
        delay(50);
        float v1 = medianFilteredVoltage(analogPins[ch]);
        diffs_mV[ch] = (v1 - vo) * 1000.0;
      }
      // apply model: glucose = -0.08*ch1 + 0.05*ch2 + 0.01*ch3
      float g = -0.08 * diffs_mV[0]
              + 0.05 * diffs_mV[1]
              + 0.01 * diffs_mV[2];
      sumModel += g;
    }
    lastGlucose = sumModel / AVG_SAMPLES;
  }
  // Determine status
  const char* status;
  if (lastGlucose < NORMAL_LOW) status = "LOW";
  else if (lastGlucose > NORMAL_HIGH) status = "HIGH";
  else status = "NORMAL";

  unsigned int secsSince = (now - lastMeasurementTime) / 1000;
  updateDisplay(lastGlucose, status, secsSince);

  delay(100);
}
