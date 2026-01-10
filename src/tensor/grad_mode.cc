/**
 *    Author:  Vishnu Satish
 *    Created: Jan 10, 2026
 **/

#include "grad_mode.h"

thread_local bool GradMode::enabled = true;

NoGrad::NoGrad() : prev{GradMode::enabled} { GradMode::enabled = false; }

NoGrad::~NoGrad() { GradMode::enabled = prev; }
