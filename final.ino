

#include "math.h"
#define DEBUG 1



int triggPin1 = 2;
int echoPin1 = 3;
int triggPin2 = 4;
int echoPin2 = 5;
int triggPin3 = 6;
int echoPin3 = 7;
int motor1=8;
int motor2=9;
int motor3=10;
int motor4=11;

/******************************************************************

Network Configuration - customized per network

******************************************************************/

const int PatternCount = 8;

const int InputNodes = 3;

const int HiddenNodes = 4;

const int OutputNodes = 4;

const float LearningRate = 0.3;

const float Momentum = 0.9;

const float InitialWeightMax = 0.5;

const float Success = 0.005;

float Input[PatternCount][InputNodes] = {

{ 0,0,0}, 
{ 0,0,1},
{ 0,1,0},
{ 0,1,1},
{ 1,0,0},
{ 1,0,1},
{ 1,1,0},
{ 1,1,1}

};

const float Target[PatternCount][OutputNodes] = {

{ 0,1,0,1 }, 
{ 0,1,0,1 },
{ 0,1,1,0 },
{ 0,1,1,0 },
{ 0,1,0,1 },
{ 0,1,0,1 },
{ 1,0,0,1 },
{ 0,0,0,0 }

};

/******************************************************************

End Network Configuration

******************************************************************/

int i, j, p, q, r;

int ReportEvery1000;

int RandomizedIndex[PatternCount];

long TrainingCycle;

float Rando;

float Error = 2;

float Accum;

float Hidden[HiddenNodes];

float Output[OutputNodes];

float HiddenWeights[InputNodes + 1][HiddenNodes];

float OutputWeights[HiddenNodes + 1][OutputNodes];

float HiddenDelta[HiddenNodes];

float OutputDelta[OutputNodes];

float ChangeHiddenWeights[InputNodes + 1][HiddenNodes];

float ChangeOutputWeights[HiddenNodes + 1][OutputNodes];

void setup() {

Serial.begin(9600);


pinMode(triggPin1, OUTPUT);
pinMode(echoPin1, INPUT);
pinMode(triggPin2, OUTPUT);
pinMode(echoPin2, INPUT);
pinMode(triggPin3, OUTPUT);
pinMode(echoPin3, INPUT);

pinMode(motor1,OUTPUT);
pinMode(motor2,OUTPUT);
pinMode(motor3,OUTPUT);
pinMode(motor4,OUTPUT);

randomSeed(analogRead(3)); //Collect a random ADC sample for Randomization.

ReportEvery1000 = 1;

for ( p = 0 ; p < PatternCount ; p++ ) {

RandomizedIndex[p] = p ;

}

Serial.println("do train_nn"); // do training neural net - only takes seconds

train_nn();

delay(1000);

}

void loop() {

//unsigned long currentMillis = millis();
testUSSensors(); // only run this to test sensors without neural net

drive_nn();

}

// just test ir sensors without neural net

void testUSSensors() {
digitalWrite(triggPin1, LOW);
delayMicroseconds(2);
digitalWrite(triggPin1, HIGH);
delayMicroseconds(10);
digitalWrite(triggPin1, LOW);
float dur1=pulseIn(echoPin1,HIGH);
float distance1=(dur1/2) / 29.1;
digitalWrite(triggPin2, LOW);
delayMicroseconds(2);
digitalWrite(triggPin2, HIGH);
delayMicroseconds(10);
digitalWrite(triggPin2, LOW);
float dur2=pulseIn(echoPin2,HIGH);
float distance2=(dur2/2) / 29.1;
digitalWrite(triggPin3, LOW);
delayMicroseconds(2);
digitalWrite(triggPin3, HIGH);
delayMicroseconds(10);
digitalWrite(triggPin3, LOW);
float dur3=pulseIn(echoPin3,HIGH);
float distance3=(dur3/2) / 29.1;
Serial.print("sensor1=");
Serial.println(distance1);
Serial.print("sensor2=");
Serial.println(distance2);
Serial.print("sensor3=");
Serial.println(distance3);

}

//USES TRAINED NEURAL NETWORK TO DRIVE ROBOT

void drive_nn()

{

Serial.println("Running NN Drive Test");

while (1) {

float TestInput[] = {0, 0, 0};
digitalWrite(triggPin1, LOW);
delayMicroseconds(2);
digitalWrite(triggPin1, HIGH);
delayMicroseconds(10);
digitalWrite(triggPin1, LOW);
float dur1=pulseIn(echoPin1,HIGH);
float distance1=(dur1/2) / 29.1;
digitalWrite(triggPin2, LOW);
delayMicroseconds(2);
digitalWrite(triggPin2, HIGH);
delayMicroseconds(10);
digitalWrite(triggPin2, LOW);
float dur2=pulseIn(echoPin2,HIGH);
float distance2=(dur2/2) / 29.1;
digitalWrite(triggPin3, LOW);
delayMicroseconds(2);
digitalWrite(triggPin3, HIGH);
delayMicroseconds(10);
digitalWrite(triggPin3, LOW);
float dur3=pulseIn(echoPin3,HIGH);
float distance3=(dur3/2) / 29.1;
int sensor1 = map(distance1, 0, 100, 0, 100); 
int sensor2 = map(distance2, 0, 100, 0, 100);
int sensor3 = map(distance3, 0, 100, 0, 100);
TestInput[0] = float(sensor1) / 100; 
TestInput[1] = float(sensor2) / 100;
TestInput[2] = float(sensor3) / 100; 


Serial.print("testinput0=");
Serial.println(TestInput[0]);
Serial.print("testinput1=");
Serial.println(TestInput[1]);
Serial.print("testinput2=");
Serial.println(TestInput[2]);

InputToOutput(TestInput[0], TestInput[1],TestInput[2]); //INPUT to ANN to obtain OUTPUT

int nnOutput1 = Output[0] * 100;

nnOutput1 = map(nnOutput1, 0, 100, 30, 140);

if (nnOutput1 != 1001) 
{
  //Serial.print("nnOutput1=");
//Serial.println(nnOutput1);


 }

delay(100);

}

}

//DISPLAYS INFORMATION WHILE TRAINING

void toTerminal()

{

for ( p = 0 ; p < PatternCount ; p++ ) {

Serial.println();

Serial.print (" Training Pattern: ");

Serial.println (p);

Serial.print (" Input ");

for ( i = 0 ; i < InputNodes ; i++ ) {

Serial.print (Input[p][i], DEC);

Serial.print (" ");

}

Serial.print (" Target ");

for ( i = 0 ; i < OutputNodes ; i++ ) {

Serial.print (Target[p][i], DEC);

Serial.print (" ");

}

/******************************************************************

Compute hidden layer activations

******************************************************************/

for ( i = 0 ; i < HiddenNodes ; i++ ) {

Accum = HiddenWeights[InputNodes][i] ;

for ( j = 0 ; j < InputNodes ; j++ ) {

Accum += Input[p][j] * HiddenWeights[j][i] ;

}

Hidden[i] = 1.0 / (1.0 + exp(-Accum)) ;

}

/******************************************************************

Compute output layer activations and calculate errors

******************************************************************/

for ( i = 0 ; i < OutputNodes ; i++ ) {

Accum = OutputWeights[HiddenNodes][i] ;

for ( j = 0 ; j < HiddenNodes ; j++ ) {

Accum += Hidden[j] * OutputWeights[j][i] ;

}

Output[i] = 1.0 / (1.0 + exp(-Accum)) ;

}

Serial.print (" Output ");
for ( i = 0 ; i < OutputNodes ; i++ ) {
Serial.print (Output[i], 5);
Serial.print (" ");

}

 }

}

void InputToOutput(float In1, float In2,float In3)

{

float TestInput[] = {0, 0, 0};

TestInput[0] = In1;

TestInput[1] = In2;

TestInput[2] = In3;

/******************************************************************

Compute hidden layer activations

******************************************************************/

for ( i = 0 ; i < HiddenNodes ; i++ ) {

Accum = HiddenWeights[InputNodes][i] ;

for ( j = 0 ; j < InputNodes ; j++ ) {

Accum += TestInput[j] * HiddenWeights[j][i] ;

}

Hidden[i] = 1.0 / (1.0 + exp(-Accum)) ;

}

/******************************************************************

Compute output layer activations and calculate errors

******************************************************************/

for ( i = 0 ; i < OutputNodes ; i++ ) {

Accum = OutputWeights[HiddenNodes][i] ;

for ( j = 0 ; j < HiddenNodes ; j++ ) {

Accum += Hidden[j] * OutputWeights[j][i] ;

}

Output[i] = 1.0 / (1.0 + exp(-Accum)) ;

}


#ifdef DEBUG

Serial.print (" Output ");

for ( i = 0 ; i < OutputNodes ; i++ ) {

Serial.print (Output[i], 5);

Serial.println (" ");

}
#endif
if(Output[0]>0.80)
{
  digitalWrite(motor1,HIGH);
}
else 
{
  digitalWrite(motor1,LOW);
  
}
if(Output[1]>=0.80)
{
  digitalWrite(motor2,HIGH);
}
else 
{
  digitalWrite(motor2,LOW);
  
}
if(Output[2]>=0.80)
{
  digitalWrite(motor3,HIGH);
}
else 
{
  digitalWrite(motor3,LOW);
  
}
if(Output[3]>0.80)
{
  digitalWrite(motor4,HIGH);
}
else 
{
  digitalWrite(motor4,LOW);
  
}

}

//TRAINS THE NEURAL NETWORK

void train_nn() {

/******************************************************************

Initialize HiddenWeights and ChangeHiddenWeights

******************************************************************/

// prog_start = 0;

Serial.println("start training...");

for ( i = 0 ; i < HiddenNodes ; i++ ) {

for ( j = 0 ; j <= InputNodes ; j++ ) {

ChangeHiddenWeights[j][i] = 0.0 ;

Rando = float(random(100)) / 100;

HiddenWeights[j][i] = 2.0 * ( Rando - 0.5 ) * InitialWeightMax ;

}

}

/******************************************************************

Initialize OutputWeights and ChangeOutputWeights

******************************************************************/

for ( i = 0 ; i < OutputNodes ; i ++ ) {

for ( j = 0 ; j <= HiddenNodes ; j++ ) {

ChangeOutputWeights[j][i] = 0.0 ;

Rando = float(random(100)) / 100;

OutputWeights[j][i] = 2.0 * ( Rando - 0.5 ) * InitialWeightMax ;

}

}

Serial.println("Initial/Untrained Outputs: ");

toTerminal();

/******************************************************************

Begin training

******************************************************************/

for ( TrainingCycle = 1 ; TrainingCycle < 2147483647 ; TrainingCycle++) {

/******************************************************************

Randomize order of training patterns

******************************************************************/

for ( p = 0 ; p < PatternCount ; p++) {

q = random(PatternCount);

r = RandomizedIndex[p] ;

RandomizedIndex[p] = RandomizedIndex[q] ;

RandomizedIndex[q] = r ;

}

Error = 0.0 ;

/******************************************************************

Cycle through each training pattern in the randomized order

******************************************************************/

for ( q = 0 ; q < PatternCount ; q++ ) {

p = RandomizedIndex[q];

/******************************************************************

Compute hidden layer activations

******************************************************************/

for ( i = 0 ; i < HiddenNodes ; i++ ) {

Accum = HiddenWeights[InputNodes][i] ;

for ( j = 0 ; j < InputNodes ; j++ ) {

Accum += Input[p][j] * HiddenWeights[j][i] ;

}

Hidden[i] = 1.0 / (1.0 + exp(-Accum)) ;

}

/******************************************************************

Compute output layer activations and calculate errors

******************************************************************/

for ( i = 0 ; i < OutputNodes ; i++ ) {

Accum = OutputWeights[HiddenNodes][i] ;

for ( j = 0 ; j < HiddenNodes ; j++ ) {

Accum += Hidden[j] * OutputWeights[j][i] ;

}

Output[i] = 1.0 / (1.0 + exp(-Accum)) ;

OutputDelta[i] = (Target[p][i] - Output[i]) * Output[i] * (1.0 - Output[i]) ;

Error += 0.5 * (Target[p][i] - Output[i]) * (Target[p][i] - Output[i]) ;

}

/******************************************************************

Backpropagate errors to hidden layer

******************************************************************/

for ( i = 0 ; i < HiddenNodes ; i++ ) {

Accum = 0.0 ;

for ( j = 0 ; j < OutputNodes ; j++ ) {

Accum += OutputWeights[i][j] * OutputDelta[j] ;

}

HiddenDelta[i] = Accum * Hidden[i] * (1.0 - Hidden[i]) ;

}

/******************************************************************

Update Inner-->Hidden Weights

******************************************************************/

for ( i = 0 ; i < HiddenNodes ; i++ ) {

ChangeHiddenWeights[InputNodes][i] = LearningRate * HiddenDelta[i] + Momentum * ChangeHiddenWeights[InputNodes][i] ;

HiddenWeights[InputNodes][i] += ChangeHiddenWeights[InputNodes][i] ;

for ( j = 0 ; j < InputNodes ; j++ ) {

ChangeHiddenWeights[j][i] = LearningRate * Input[p][j] * HiddenDelta[i] + Momentum * ChangeHiddenWeights[j][i];

HiddenWeights[j][i] += ChangeHiddenWeights[j][i] ;

}

}

/******************************************************************

Update Hidden-->Output Weights

******************************************************************/

for ( i = 0 ; i < OutputNodes ; i ++ ) 
{

ChangeOutputWeights[HiddenNodes][i] = LearningRate * OutputDelta[i] + Momentum * ChangeOutputWeights[HiddenNodes][i] ;

OutputWeights[HiddenNodes][i] += ChangeOutputWeights[HiddenNodes][i] ;

for ( j = 0 ; j < HiddenNodes ; j++ ) {

ChangeOutputWeights[j][i] = LearningRate * Hidden[j] * OutputDelta[i] + Momentum * ChangeOutputWeights[j][i] ;

OutputWeights[j][i] += ChangeOutputWeights[j][i] ;

}

}

}

/******************************************************************

Every 100 cycles send data to terminal for display and draws the graph on OLED

******************************************************************/

ReportEvery1000 = ReportEvery1000 - 1;

if (ReportEvery1000 == 0)

{

Serial.print ("TrainingCycle: ");

Serial.print (TrainingCycle);

Serial.print (" Error = ");

Serial.println (Error, 5);

toTerminal();

if (TrainingCycle == 1)

{

ReportEvery1000 = 99;

}

else

{

ReportEvery1000 = 100;

}

}

/******************************************************************

If error rate is less than pre-determined threshold then end

******************************************************************/

if ( Error < Success ) break ;

}

}


