# Joy Ride - Project Overview
This is a quick, simple project that gives you a chance to write code that controls a simulated car as you get familiar with the Workspaces you'll be using throughout this Nanodegree program. This project has three parts (but you will only submit part three).

## Part One - Drag Race: 
In this part, you'll write code that lets a car jump over a grove of trees. This might not be a common scenario for a self-driving car, but it will get you familiar with the programming interface.
## Part Two - Circular Track: 
In this part, you'll write code that lets a car navigate a circular track. In doing so you'll explore the relationship between steering angle and turning radius.
## Part Three - Parallel Park: 
In this part, you'll write a sequence of instructions that successfully parallel parks a car. In a real car, you obviously wouldn't be able to try repeatedly and hit other cars while you perfect your parking, but in this project, you can try as many times as you need to!

## Tip
For Part Three, the simulator contains a little bit of noise each time you launch it. To help account for this, consider spending a second or two of time traveling slowly forward in the simulator before reversing into your parallel parking. You should also consider using a fairly low vehicle velocity at all times.

## Reflection
Note: If you didn't already, make sure you submit part three by clicking the SUBMIT button in the lower of the Part 3 notebook.

The activities you just went through were simplified: when designing a self driving car we try to avoid writing special code that only applies to specific situations like staying on a circular track of some particular size. We prefer to write code that generalizes to many driving situations.

In order to do that, the car needs to know what's going on around it and it does that by using its sensors. Unfortunately the measurements these sensors make are UNAVOIDABLY unreliable. In the next part of this Nanodegree you'll learn probabilistic techniques for handling this unreliability.

## A note on Control Noise
While solving this project you may have noticed something strange: sometimes the same code didn't actually do the same thing!

This "bug" should help you understand what it feels like to try to program a self driving car or any system where control noise is present. A car is an imperfect mechanical system and that means that it's response to a fixed input can change over time. Near the end of this curriculum you will learn how we use motion controllers to respond to unexpected behavior from the car
