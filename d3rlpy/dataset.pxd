from libcpp.vector cimport vector
from libcpp.memory cimport make_shared, shared_ptr


cdef extern from "d3rlpy/dataset.h" namespace "d3rlpy" nogil:
    cdef cppclass CTransition[T]:
        vector[int] observation_shape
        int action_size
        T* observation
        float reward
        T* next_observation
        float next_reward
        float terminal
        shared_ptr[CTransition[T]] prev_transition
        shared_ptr[CTransition[T]] next_transition
