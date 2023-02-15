#define NO_INFECTADO 0
#define INFECTADO 1
#define RECUPERADO 2

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
  unsigned int iteracionesCercaDeUnInfectado;
  unsigned int iteracionesInfectado;
};

// https://stackoverflow.com/questions/9912143/how-to-get-a-random-number-in-opencl
// Regresa un valor entre 0 y 1
static float simpleNoise(float x, float y, float z) {
    float ptr = 0.0f;
    return fract(sin(x*112.9898f + y*179.233f + z*237.212f) * 43758.5453f, &ptr);
}

// Actualizar el buffer de la informacion de los agentes
__kernel void count_infected_agents(__global struct Agent *listaDeAgentes, __global unsigned int *infectedCount, unsigned int iteration) {
  int gid = get_global_id(0);

  if (listaDeAgentes[gid].estado == INFECTADO) {
    atomic_add(&infectedCount[iteration], (unsigned int) 1);
  }
}

__kernel void update_agent(__global struct Agent *listaDeAgentes, const unsigned int numeroDeAgentes, unsigned int iteration, float borderX, float borderY) {
  // Debido a como funciona julia, las iteraciones comienzan en 1, gracias a esto, y considerando
  // que nuestro array si comienza en '0' (segun opencl), en la primera iteracion (iteration = 1)
  // ya vamos a tener el estado inicial en nuestro index de listaDeAgentes 0

  // Necesitamos las posiciones de las iteraciones anteriores para calcular las nuevas
  int gid = get_global_id(0) + (iteration * numeroDeAgentes);

  int prev_gid = get_global_id(0) + ((iteration - 1) * numeroDeAgentes);

  listaDeAgentes[gid] = listaDeAgentes[prev_gid];

  const float minimoRadiodeInfeccion = 5;
  bool almenosUnInfectadoCerca = false;

  // checar si esta agente esta cerca de un infectado
  if (listaDeAgentes[gid].estado == NO_INFECTADO) {
    int prev_iteration = iteration * numeroDeAgentes;

    for (int i = 0; i < numeroDeAgentes; i++) {
      int real_i = (iteration * numeroDeAgentes) + i;

      if (real_i != prev_gid &&
        // checar si este otro agente esta infectado
        listaDeAgentes[real_i].estado == INFECTADO &&
        // checar la distancia con el otro agente
        sqrt(pow(listaDeAgentes[prev_gid].pos.x - listaDeAgentes[real_i].pos.x, 2) + pow(listaDeAgentes[prev_gid].pos.y - listaDeAgentes[real_i].pos.y, 2)) < minimoRadiodeInfeccion
      ) {

        listaDeAgentes[gid].iteracionesCercaDeUnInfectado += 1;
        almenosUnInfectadoCerca = true;
        break;
      }
    }

  // Llevar la cuenta de cuanto tiempo lleva infectado cada agente
  } else if (listaDeAgentes[gid].estado == INFECTADO) {
    listaDeAgentes[gid].iteracionesInfectado += 1;
  }

  // Los agentes se recuperan despues de un cierto tiempo
  if (listaDeAgentes[gid].iteracionesInfectado > 20) {
    listaDeAgentes[gid].estado = RECUPERADO;
  }

  // Si no encuentra un agente infectado en su vicinidad se resettea el contador
  // no estoy seguro si deberia ser asi
  // if (!almenosUnInfectadoCerca) {
  //   listaDeAgentes[gid].iteracionesCercaDeUnInfectado = 0;
  // }

  float p = 0.12;
  // decidir, en base a una probabilidad random y si a pasado suficente
  // tiempo cerca de un infectado, si deberia considerarse infectado
  if (
    (listaDeAgentes[prev_gid].iteracionesCercaDeUnInfectado > 15) &&
    // el valor p se podria multiplicar por listaDeAgentes[gid].iteracionesCercaDeUnInfectado para
    // hacer que dependa de la cantidad de tiempo que ha pasado junto a otros infectados
    (simpleNoise((float) gid, (float) iteration, (float) gid / (float) iteration) > p) && 
    (listaDeAgentes[prev_gid].estado == NO_INFECTADO)
  ) {
    listaDeAgentes[gid].estado = INFECTADO;
  }

  float epsilon = 0.0001;
  struct Vec2 nuevaPosicion = {listaDeAgentes[prev_gid].pos.x + listaDeAgentes[prev_gid].vel.x, listaDeAgentes[prev_gid].pos.y + listaDeAgentes[prev_gid].vel.y};

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
