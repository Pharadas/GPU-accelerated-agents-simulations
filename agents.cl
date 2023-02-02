struct Vec2 {
  float x;
  float y;
};

struct Agent {
  struct Vec2 pos;
  struct Vec2 vel;
  uchar state;
  unsigned int iterations_spent_next_to_infected;
  unsigned int iterations_spent_infected;
};

// https://stackoverflow.com/questions/9912143/how-to-get-a-random-number-in-opencl
// Return value between 0 and 1
static float simpleNoise(float x, float y, float z) {
    float ptr = 0.0f;
    return fract(sin(x*112.9898f + y*179.233f + z*237.212f) * 43758.5453f, &ptr);
}

__kernel void clear_cuadrants(__global unsigned int *cuadrants, __global unsigned int *cuadrantsFilled, unsigned int maxAgentsPerCuadrant) {
  int gid = get_global_id(0);

  cuadrantsFilled[gid] = 0;
  for (int i = 0; i < maxAgentsPerCuadrant; i++) {
    cuadrants[(gid * maxAgentsPerCuadrant) + i] = 0;
  }
}

// No se utiliza por ahora
__kernel void update_cuadrants(__global struct Agent *agentsList,
                               __global unsigned int *cuadrants,
                               __global unsigned int *cuadrants_filled,
                               unsigned int max_agents_per_cuadrant,
                               unsigned int sizeOfCuadrant) {
  int gid = get_global_id(0);
  struct Vec2 border = {1000., 1000.};

  struct Agent workingAgent = agentsList[gid];

  // encontrar en que cuadrante se encuentra este agente
  if (workingAgent.pos.x == 0.) {
    workingAgent.pos.x = 0.1;
  }

  if (workingAgent.pos.y == 0.) {
    workingAgent.pos.y = 0.1;
  }

  struct Vec2 cuadrant = {trunc(workingAgent.pos.x / sizeOfCuadrant), trunc(workingAgent.pos.y / sizeOfCuadrant)};
  int linearCuadrant = (convert_int(cuadrant.x * trunc(border.x / sizeOfCuadrant))) + convert_int(cuadrant.y);

  atomic_add(&cuadrants_filled[linearCuadrant], (unsigned int) 1);
}

__kernel void count_infected_agents(__global struct Agent *agentsList, __global unsigned int *infectedCount, unsigned int iteration) {
  int gid = get_global_id(0);
  if (agentsList[gid].state == 1) {
    atomic_add(&infectedCount[iteration], (unsigned int) 1);
  }
}

__kernel void update_agent(__global struct Agent *agentsList, const unsigned int numberOfAgents, unsigned int iteration, float borderX, float borderY) {
  int gid = get_global_id(0);
  const float minimumRadiusToInfect = 5;
  bool atLeastOneFound = false;

  // checar si se infecta por algun otro agente
  if (agentsList[gid].state == 0) {
    for (int i = 0; i < numberOfAgents; i++) {
      if (i != gid &&
        // checar si este otro agente esta infectado
        agentsList[i].state == 1 &&
        // checar la distancia con el otro agente
        sqrt(pow(agentsList[gid].pos.x - agentsList[i].pos.x, 2) + pow(agentsList[gid].pos.y - agentsList[i].pos.y, 2)) < minimumRadiusToInfect
      ) {
        agentsList[gid].iterations_spent_next_to_infected += 1;
        atLeastOneFound = true;
        break;
      }
    }
  } else if (agentsList[gid].state == 1) {
    agentsList[gid].iterations_spent_infected += 1;
  }

  if (agentsList[gid].iterations_spent_infected > 20) {
    agentsList[gid].state = 2;
  }

  // Si no encuentra un agente infectado en su vicinidad se resettea el contador
  // no estoy seguro si deberia ser asi
  // if (!atLeastOneFound) {
  //   agentsList[gid].iterations_spent_next_to_infected = 0;
  // }

  float p = 0.5;
  if (
    (agentsList[gid].iterations_spent_next_to_infected > 15) &&
    // el valor p se podria multiplicar por agentsList[gid].iterations_spent_next_to_infected para
    // hacer que dependa de la cantidad de tiempo que ha pasado junto a otros infectados
    // (simpleNoise((float) gid, (float) iteration, (float) gid / (float) iteration) > p)
    (agentsList[gid].state == 0)
  ) {
    agentsList[gid].state = 1;
  }

  float epsilon = 0.0001;

  struct Vec2 tentativePosition = {agentsList[gid].pos.x + agentsList[gid].vel.x, agentsList[gid].pos.y + agentsList[gid].vel.y};

  // El operador modulo tiene una implementacion diferente, A % B, cuando A es un numero negativo, da A
  if (tentativePosition.x >= borderX - epsilon) {
    tentativePosition.x = epsilon;
  }

  if (tentativePosition.y >= borderY - epsilon) {
    tentativePosition.y = epsilon;
  }

  if (tentativePosition.x <= 0) {
    tentativePosition.x = borderX - epsilon;
  }

  if (tentativePosition.y <= 0) {
    tentativePosition.y = borderY - epsilon;
  }

  agentsList[gid].pos = tentativePosition;
}
