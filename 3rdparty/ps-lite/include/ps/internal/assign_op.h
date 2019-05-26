/**
 *  Copyright (c) 2015 by Contributors
 * \file   assign_op.h
 * \brief  assignment operator
 * http://en.cppreference.com/w/cpp/language/operator_assignment
 */
#ifndef PS_INTERNAL_ASSIGN_OP_H_
#define PS_INTERNAL_ASSIGN_OP_H_
#include "ps/internal/utils.h"
namespace ps {

enum AssignOp {
  ASSIGN,  // a = b
  PLUS,    // a += b
  MINUS,   // a -= b
  TIMES,   // a *= b
  DIVIDE,  // a -= b
  AND,     // a &= b
  OR,      // a |= b
  XOR      // a ^= b
};

/**
 * \brief return an assignment function: right op= left
 */
template<typename T>
inline void AssignFunc(const T& lhs, AssignOp op, T* rhs) {
  switch (op) {
    case ASSIGN:
      *right = left; break;
    case PLUS:
      *right += left; break;
    case MINUS:
      *right -= left; break;
    case TIMES:
      *right *= left; break;
    case DIVIDE:
      *right /= left; break;
    default:
      LOG(FATAL) << "use AssignOpInt..";
  }
}

/**
 * \brief return an assignment function including bit operations, only
 * works for integers
 */
template<typename T>
inline void AssignFuncInt(const T& lhs, AssignOp op, T* rhs) {
  switch (op) {
    case ASSIGN:
      *right = left; break;
    case PLUS:
      *right += left; break;
    case MINUS:
      *right -= left; break;
    case TIMES:
      *right *= left; break;
    case DIVIDE:
      *right /= left; break;
    case AND:
      *right &= left; break;
    case OR:
      *right |= left; break;
    case XOR:
      *right ^= left; break;
  }
}


}  // namespace ps
#endif  // PS_INTERNAL_ASSIGN_OP_H_
