/* A set of quick-fire tests */

const int IR1 = 5; // IR sensors pins
const int IR2 = 18;
const int IR3 = 32;
const int IR4 = 33;
const int IR5 = 25;

// looking from the back

const int motor1f = 27; // motor LEFT forward
const int motor1r = 13; // motor LEFT reverse

const int motor2f = 12; // motor RIGHT forward
const int motor2r = 14; // motor RIGHT reverse

const int led_red = 2; // misc
const int led_green = 15;
const int buzzer = 23;

void setup()
{
    pinMode(IR1, INPUT);
    pinMode(IR2, INPUT);
    pinMode(IR3, INPUT);
    pinMode(IR4, INPUT);
    pinMode(IR5, INPUT);

    pinMode(motor1f, OUTPUT);
    pinMode(motor1r, OUTPUT);
    pinMode(motor2f, OUTPUT);
    pinMode(motor2r, OUTPUT);

    pinMode(led_red, OUTPUT);
    pinMode(led_green, OUTPUT);

    pinMode(buzzer, OUTPUT);

    digitalWrite(buzzer, HIGH); // buzzer OFF is on HIGH

    digitalWrite(led_red, HIGH);

    Serial.begin(115200);
}

void buzzerTest()
{
    digitalWrite(buzzer, LOW);
    delay(1000);
    digitalWrite(buzzer, HIGH);
}

void ledsTest()
{
    Serial.println("led");
    // Serial.println(millis());
    digitalWrite(led_red, HIGH);
    // digitalWrite(led_green, HIGH);
    // delay(1000);
    // unsigned long start = millis();
    // unsigned long diff = 0;
    delay(1000);
    digitalWrite(led_red, LOW);
    // digitalWrite(led_green, LOW);
    Serial.println("led end");
    // Serial.println(millis());
}

void loop()
{
    // buzzerTest();
    // ledsTest();
    // readIRs();
    // teleop();
    // readIRs();
    // motorTest();
    // Serial.println("hey babes");
    // buzzerTest();
    // ledsTest();
    // digitalWrite(led_red, LOW);
    delay(1000);
    digitalWrite(led_red, LOW);
    delay(1000);
    digitalWrite(led_red, HIGH);
    Serial.println("buzz buzz beeee");
    }
