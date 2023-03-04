#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

// #define RUN_AWAY

// Los susceptibles van a usar su contador
// para saber si se deberian infectar o no
#define SUSCEPTIBLE 0

// Los asintomaticos tienen una probabilidad pequena
// de pasar al estado EN_CUARENTENA, pero si terminan
// su tiempo de enfermedad se convierten en recuperados
#define INFECTADO_ASINTOMATICO 1

// los sintomaticos tienen una probabilidad
// mas alta de irse a cuarentena, pero cuando
// terminen sus iteraciones como infectados
// pasaran a ser recuperados
#define INFECTADO_SINTOMATICO 2

// los recuperados no interactuan con ningun
// otro agente, solo se mueven, si su contador
// termina, se vuelven susceptibles
#define RECUPERADO 3

// los agentes en cuarentena no se mueven y no infectan
// a los otros agentes, cuando su contador termina
// se vuelven recuperados
#define EN_CUARENTENA 4

#define LOCKED 0
#define UNLOCKED 1

// Es importante que estos dos structs sean iguales que
// en la cpu
// Los floats son, por default
// de 32 bits
struct Vec2 {
  float x;
  float y;
};

struct Agent {
  struct Vec2 pos;
  struct Vec2 vel;
  // Uso un uchar porque es la manera mas facil de lidiar con un objeto de 8 bytes
  uchar estado;
  uchar counter;
};

// https://stackoverflow.com/questions/9912143/how-to-get-a-random-number-in-opencl
// Regresa un valor entre 0 y 1
static float simpleNoise(float x, float y, float z) {
    float ptr = 0.0f;
    return fract(sin(x*112.9898f + y*179.233f + z*237.212f) * 43758.5453f, &ptr);
}

__kernel void clear_buffer(
  __global unsigned int *buffer
) {
  int gid = get_global_id(0);

  buffer[gid] = 0;
}

__kernel void assign_agents_to_cuadrants(
  __global struct Agent *listaDeAgentes,
  __global unsigned int *cuadrants,
  __global unsigned int *filled_cuadrants,
  __global unsigned int *cuadrants_counter,
  __global unsigned int *cuadrants_semaphores,
  unsigned int numeroDeAgentes,
  unsigned int iteration,
  unsigned int sizeOfCuadrant,
  float borderX, 
  float borderY
) {
  int prev_gid = get_global_id(0) + ((iteration - 1) * numeroDeAgentes);

  struct Agent workingAgent = listaDeAgentes[prev_gid];

  if (workingAgent.pos.x == 0.) {
    workingAgent.pos.x = 0.001;
  }

  if (workingAgent.pos.y == 0.) {
    workingAgent.pos.y = 0.001;
  }

  struct Vec2 cuadrantCoord = {trunc((workingAgent.pos.x) / sizeOfCuadrant), trunc((workingAgent.pos.y) / sizeOfCuadrant)};
  int linearCuadrant = convert_int(cuadrantCoord.x) + convert_int(cuadrantCoord.y * trunc(borderY / sizeOfCuadrant));

  while (atomic_cmpxchg(&cuadrants_semaphores[linearCuadrant], UNLOCKED, LOCKED) != UNLOCKED) {
    cuadrants[filled_cuadrants[linearCuadrant] + cuadrants_counter[linearCuadrant]] = get_global_id(0);
    cuadrants_counter[linearCuadrant] = cuadrants_counter[linearCuadrant] + 1;
    cuadrants_semaphores[linearCuadrant] = UNLOCKED;
  }
}

__kernel void update_filled_cuadrants(
  __global struct Agent *agentsList,
  __global unsigned int *cuadrants,
  __global unsigned int *cuadrantsFilled,
  unsigned int iteration,
  unsigned int numeroDeAgentes,
  unsigned int sizeOfCuadrant,
  float borderX, 
  float borderY
) {
  int prev_gid = get_global_id(0) + ((iteration - 1) * numeroDeAgentes);
  // // // int prev_gid = get_global_id(0);

  struct Agent workingAgent = agentsList[prev_gid];

  if (workingAgent.pos.x == 0.) {
    workingAgent.pos.x = 0.001;
  }

  if (workingAgent.pos.y == 0.) {
    workingAgent.pos.y = 0.001;
  }

  struct Vec2 cuadrantCoord = {trunc((workingAgent.pos.x) / sizeOfCuadrant), trunc((workingAgent.pos.y) / sizeOfCuadrant)};
  int linearCuadrant = convert_int(cuadrantCoord.x) + convert_int(cuadrantCoord.y * trunc(borderY / sizeOfCuadrant));

  // agentsList[prev_gid].iteracionesCercaDeUnInfectado = linearCuadrant;

  atomic_inc(&(cuadrantsFilled[linearCuadrant]));
}

__kernel void calcular_posiciones_de_cuadriculas(
  __global unsigned int *filled_cuadrants,
  unsigned int amount_of_grid_spaces
) {
  unsigned int last_val = 0;

  for (int i = 0; i < amount_of_grid_spaces; i++) {
    unsigned int r = filled_cuadrants[i];
    filled_cuadrants[i] = last_val;
    last_val = r + last_val;
  }
}

__kernel void update_agent(
  __global struct Agent *listaDeAgentes, 
  __global unsigned int *cuadrants,
  __global unsigned int *filled_cuadrants,
  const unsigned int numeroDeAgentes, 
  unsigned int iteration, 
  unsigned int sizeOfCuadrant,
  unsigned int amount_of_cuadrants,
  float borderX,
  float borderY
) {
  // Debido a como funciona julia las iteraciones comienzan en 1, gracias a esto y considerando
  // que nuestro array comienza en 0 en OpenCL, en la primera iteracion (iteration = 1)
  // ya vamos a tener el estado inicial en nuestro index de listaDeAgentes 0

  // Necesitamos las posiciones de las iteraciones anteriores para calcular las nuevas
  int g = get_global_id(0);
  int gid = g + (iteration * numeroDeAgentes);
  int prev_gid = g + ((iteration - 1) * numeroDeAgentes);

  const float probabilidad_susceptible_a_asintomatico = 0.5;



  listaDeAgentes[gid] = listaDeAgentes[prev_gid];

  const float minimoRadiodeInfeccion = 5;
  bool almenosUnInfectadoCerca = false;

  struct Agent workingAgent = listaDeAgentes[prev_gid];

  if (workingAgent.pos.x == 0.) {
    workingAgent.pos.x = 0.1;
  }

  if (workingAgent.pos.y == 0.) {
    workingAgent.pos.y = 0.1;
  }

  struct Vec2 cuadrantCoord = {trunc((workingAgent.pos.x) / sizeOfCuadrant), trunc((workingAgent.pos.y) / sizeOfCuadrant)};
  int ttt = trunc(borderY / sizeOfCuadrant);

  if (listaDeAgentes[gid].estado == SUSCEPTIBLE) {
    #ifdef RUN_AWAY
      struct Vec2 fuerzaDeEmpuje = {0., 0.};
    #endif

    if ((listaDeAgentes[gid].counter > 10)) {
      if (simpleNoise(listaDeAgentes[gid].pos.x, listaDeAgentes[gid].pos.y, (float) iteration) > probabilidad_susceptible_a_asintomatico) {
        listaDeAgentes[gid].estado = INFECTADO_ASINTOMATICO;
      } else {
        listaDeAgentes[gid].estado = INFECTADO_ASINTOMATICO;
      }
      listaDeAgentes[gid].counter = 0;

    } else {
      // vamos a checar los sectores de alrededor
      for (int xSector = cuadrantCoord.x - 1; xSector <= cuadrantCoord.x + 1; xSector++) {
        for (int ySector = cuadrantCoord.y - 1; ySector <= cuadrantCoord.y + 1; ySector++) {
          int linearCuadrant = convert_int(xSector) + convert_int(ySector * ttt);
          // Hay que asegurarnos que no nos vayamos a salir del array
          // e intentemos acceder memoria que no nos pertenece
          if (linearCuadrant >= 0 && linearCuadrant <= amount_of_cuadrants) {
            // buscar que haya algun otro agente en el mismo cuadrante
            unsigned int agents_in_section = 0;

            if (linearCuadrant == amount_of_cuadrants) {
              agents_in_section = numeroDeAgentes - filled_cuadrants[linearCuadrant];
            } else {
              agents_in_section = filled_cuadrants[linearCuadrant + 1] - filled_cuadrants[linearCuadrant];
            }

            for (int i = filled_cuadrants[linearCuadrant]; i < filled_cuadrants[linearCuadrant] + agents_in_section; i++) {
              long int r = cuadrants[i] + ((iteration - 1) * numeroDeAgentes);
              float dist = (pow(listaDeAgentes[gid].pos.x - listaDeAgentes[r].pos.x, 2) + pow(listaDeAgentes[gid].pos.y - listaDeAgentes[r].pos.y, 2));

              if (
                (i != g) &&
                ((listaDeAgentes[r].estado == INFECTADO_ASINTOMATICO) || (listaDeAgentes[r].estado == INFECTADO_SINTOMATICO))) {
                if (dist < (minimoRadiodeInfeccion * minimoRadiodeInfeccion)) {
                  almenosUnInfectadoCerca = true;
                }

                #ifdef RUN_AWAY
                  fuerzaDeEmpuje.x += (1 / dist) * (listaDeAgentes[gid].pos.x - listaDeAgentes[r].pos.x);
                  fuerzaDeEmpuje.y += (1 / dist) * (listaDeAgentes[gid].pos.y - listaDeAgentes[r].pos.y);
                #endif
              }
            }
          }
        }
      }

      if (almenosUnInfectadoCerca) {
        listaDeAgentes[gid].counter = listaDeAgentes[gid].counter + 1;

        #ifdef RUN_AWAY
          // normalize force
          float totalForce = sqrt(pow(fuerzaDeEmpuje.x, 2) + pow(fuerzaDeEmpuje.y, 2));
          fuerzaDeEmpuje.x = (fuerzaDeEmpuje.x / totalForce) * 2;
          fuerzaDeEmpuje.y = (fuerzaDeEmpuje.y / totalForce) * 2;

          listaDeAgentes[gid].vel.x += fuerzaDeEmpuje.x;
          listaDeAgentes[gid].vel.y += fuerzaDeEmpuje.y;
        #endif
      }
    }

  } else if (listaDeAgentes[gid].estado == INFECTADO_SINTOMATICO) {
    // despues de cierto tiempo, existe la posibilidad de que
    // el infectado se empiece a aislar y cambie su estado
    // a EN_CUARENTENA
    if ((listaDeAgentes[gid].counter > 15) && (simpleNoise(listaDeAgentes[gid].pos.x, listaDeAgentes[gid].pos.y, (float) iteration) > 0.7)) {
      listaDeAgentes[gid].estado = EN_CUARENTENA;
    }

    // si no se aisla simplemente
    // espera a que la enfermedad pase
    if (listaDeAgentes[gid].counter > 25) {
      listaDeAgentes[gid].estado = RECUPERADO;
    } else {
      listaDeAgentes[gid].counter = listaDeAgentes[gid].counter + 1;
    }

    // los agentes en cuarentena no interactuan con los
    // demas agentes, solo esperan a que pase la enfermedad
  } else if (listaDeAgentes[gid].estado == INFECTADO_ASINTOMATICO) {
    // despues de cierto tiempo, existe la posibilidad de que
    // el infectado se empiece a aislar y cambie su estado
    // a EN_CUARENTENA
    if ((listaDeAgentes[gid].counter > 15) && (simpleNoise(listaDeAgentes[gid].pos.x, listaDeAgentes[gid].pos.y, (float) iteration) > 0.9)) {
      listaDeAgentes[gid].estado = EN_CUARENTENA;
    }

    // si no se aisla simplemente
    // espera a que la enfermedad pase
    if (listaDeAgentes[gid].counter > 25) {
      listaDeAgentes[gid].estado = RECUPERADO;
    } else {
      listaDeAgentes[gid].counter = listaDeAgentes[gid].counter + 1;
    }

    // los agentes en cuarentena no interactuan con los
    // demas agentes, solo esperan a que pase la enfermedad
  } else if (listaDeAgentes[gid].estado == EN_CUARENTENA) {
    if (listaDeAgentes[gid].counter > 20) {
      // si pasan la enfermedad, entonces se les considera
      // recuperados
      listaDeAgentes[gid].estado = RECUPERADO;
      listaDeAgentes[gid].counter = 0;
    } else {
      listaDeAgentes[gid].counter = listaDeAgentes[gid].counter + 1;
    }

  } else if (listaDeAgentes[gid].estado == RECUPERADO) {
    if (listaDeAgentes[gid].counter > 20) {
      listaDeAgentes[gid].estado = SUSCEPTIBLE;
      listaDeAgentes[gid].counter = 0;

    } else {
      listaDeAgentes[gid].counter += 1;
    }
  }

  float epsilon = 0.01;

  if (listaDeAgentes[gid].vel.x != 0. && listaDeAgentes[gid].vel.y != 0.) {
    // normalize speed
    listaDeAgentes[gid].vel.x += (simpleNoise(listaDeAgentes[gid].vel.x, listaDeAgentes[gid].vel.y, iteration) - 0.5) * 2;
    listaDeAgentes[gid].vel.y += (simpleNoise(listaDeAgentes[gid].vel.y, listaDeAgentes[gid].vel.x, iteration) - 0.5) * 2;

    float totalSpeed = sqrt(pow(listaDeAgentes[gid].vel.x, 2) + pow(listaDeAgentes[gid].vel.y, 2));
    listaDeAgentes[gid].vel.x = (listaDeAgentes[gid].vel.x / totalSpeed);
    listaDeAgentes[gid].vel.y = (listaDeAgentes[gid].vel.y / totalSpeed);
  }

  struct Vec2 nuevaPosicion = {listaDeAgentes[gid].pos.x + listaDeAgentes[gid].vel.x, listaDeAgentes[gid].pos.y + listaDeAgentes[gid].vel.y};
  if (listaDeAgentes[gid].estado == EN_CUARENTENA) {
    nuevaPosicion = listaDeAgentes[gid].pos;
  }

  // El operador modulo tiene una implementacion diferente, A % B, cuando A es un numero negativo, da A
  if (nuevaPosicion.x >= borderX - epsilon) {
    nuevaPosicion.x = epsilon;
  }

  if (nuevaPosicion.y >= borderY - epsilon) {
    nuevaPosicion.y = epsilon;
  }

  if (nuevaPosicion.x <= 0) {
    nuevaPosicion.x = borderX - epsilon;
  }

  if (nuevaPosicion.y <= 0) {
    nuevaPosicion.y = borderY - epsilon;
  }

  listaDeAgentes[gid].pos = nuevaPosicion;
}