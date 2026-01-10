/**
 *    Author:  Vishnu Satish
 *    Created: Jan 10, 2026
 **/

#pragma once

struct GradMode {
  // `enabled` has one copy per-thread.
  static thread_local bool enabled;
};

struct NoGrad {
  bool prev;

  NoGrad();
  ~NoGrad();
};
