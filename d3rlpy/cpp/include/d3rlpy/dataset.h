#include <vector>
#include <memory>

namespace d3rlpy {

  using namespace std;

  template<typename T>
  struct CTransition {
    vector<int> observation_shape;
    int action_size;
    T* observation;
    float reward;
    T* next_observation;
    float next_reward;
    float terminal;
    shared_ptr<CTransition> prev_transition;
    shared_ptr<CTransition> next_transition;
  };

}
