using Plots, Colors

const HORAS_EN_EL_DIA = 24
const CANTIDAD_DE_ITERACIONES = 100
const HORAS_DE_SIMULACION = 15
const TIEMPO_POR_ITERACION = HORAS_DE_SIMULACION / CANTIDAD_DE_ITERACIONES
const CANTIDAD_DE_AGENTES = 100
const CANTIDAD_DE_UBICACIONES = 200

const NO_INFECTADO = 0;
const INFECTADO = 1;

struct Agente
  estado::UInt8
  ubicacion::UInt8
end

function actualizar_posiciones_de_agentes(horarios, iteracion, agentes)
  for i in 1:CANTIDAD_DE_AGENTES
    agentes[i] = Agente(agentes[i].estado, horarios[iteracion, i])
  end
end

function mostrar_resultados(agentes, agentes_ordenados, i, cantidad_de_agentes_por_habitacion, ubicaciones)
  print("iteracion ", i, ":\n")
  for x in 1:CANTIDAD_DE_AGENTES
    este_agente = agentes[agentes_ordenados[x]]
    print("agente #", agentes_ordenados[x], " habitacion #", este_agente.ubicacion)
    if este_agente.estado == NO_INFECTADO
      print(" | NO INFECTADO |")
    else
      print(" | INFECTADO |")
    end
    print(" PROBABILIDAD EN BASE A SU HABITACION: ", (cantidad_de_agentes_por_habitacion[este_agente.ubicacion] / ubicaciones[este_agente.ubicacion]) * 0.1, "\n")
  end
end

function infectado(tamano_de_habitacion, cantidad_de_infectados_en_habitacion)
  # se puede cambiar por el modelo especifico de probabilidad de infeccion,
  # pero lo importante es que depende de la cantidad de infectados y el 
  # tamano de la habitacion en la que estan
  return (cantidad_de_infectados_en_habitacion / tamano_de_habitacion) * 0.1 > rand() 
end

# todavia no lo grafica bien
# tengo que avanzarle un poco a esto
function graficar_resultados(agentes, y, colores)
  xs = []
  ys = []
  curr_colores = []

  for r in y
    if agentes[r].ubicacion > 5
      push!(xs, rand() - 5 + agentes[r].ubicacion / 2)
      push!(ys, rand())
    else
      push!(xs, rand() + agentes[r].ubicacion / 2)
      push!(ys, rand() + 1)
    end
    push!(curr_colores, colores[agentes[r].ubicacion])
  end

  scatter(xs, ys, c=curr_colores, xlims = [0, 10], ylims = [0, 2]) 
end

# para que el codigo sea un poco mas facil de pasar a la GPU
# la voy a definir aqui
# Esta funcion guarda en un array la cantidad de agentes que van a haber en cada 
# habitacion
function asignar_inicios_de_habitaciones(agentes)
  cantidad_de_agentes_por_habitacion = zeros(Int, CANTIDAD_DE_UBICACIONES)
  cantidad_de_agentes_infectados_por_habitacion = zeros(Int, CANTIDAD_DE_UBICACIONES)

  for i in 1:CANTIDAD_DE_AGENTES
    cantidad_de_agentes_por_habitacion[agentes[i].ubicacion] += 1 # en la gpu esta deberia ser una atomic operation
    if agentes[i].estado == INFECTADO
      cantidad_de_agentes_infectados_por_habitacion[agentes[i].ubicacion] += 1 # en la gpu esta deberia ser una atomic operation
    end
  end

  indices_de_inicios_de_ubicaciones = zeros(Int, CANTIDAD_DE_UBICACIONES)
  curr_sum = 1
  for i in 1:CANTIDAD_DE_UBICACIONES
    indices_de_inicios_de_ubicaciones[i] = curr_sum
    curr_sum += cantidad_de_agentes_por_habitacion[i]
  end

  return (indices_de_inicios_de_ubicaciones, cantidad_de_agentes_por_habitacion, cantidad_de_agentes_infectados_por_habitacion)
end

function crear_lista_de_ubicaciones(agentes_por_ubicacion, agentes)
  agentes_ordenados = zeros(Int, CANTIDAD_DE_AGENTES)
  ubicaciones_contador = zeros(Int, CANTIDAD_DE_UBICACIONES)

  for i in 1:CANTIDAD_DE_AGENTES
    ubicacion_de_agente_actual = agentes[i].ubicacion

    agentes_ordenados[ubicaciones_contador[ubicacion_de_agente_actual] + agentes_por_ubicacion[ubicacion_de_agente_actual]] = i
    ubicaciones_contador[agentes[i].ubicacion] += 1
  end

  return agentes_ordenados
end

# las ubicaciones estan definidas solamente por sus tamanos
const ubicaciones = Array{Float32, 1}(rand(Float32, CANTIDAD_DE_UBICACIONES) .* 10 .+ 10)

# la lista de horarios sera tan larga como la cantidad de iteraciones que se haran por dia
# y en cada iteracion se guardara un indice de la ubicacion en la que deberia estar ese agente
# y hay tantos horarios como agentes, cada agente tiene una ID que es su indice en la
# lista global de agentes, que se define en la siguiente linea, usaran este indice para
# encontrar su horario en esta lista
const horarios = rand(
  UnitRange{UInt8}(1, CANTIDAD_DE_UBICACIONES), 
  (trunc(Int, HORAS_DE_SIMULACION / TIEMPO_POR_ITERACION), CANTIDAD_DE_AGENTES)
)
agentes = Array{Agente, 1}(undef, CANTIDAD_DE_AGENTES)

agentes[1] = Agente(INFECTADO, horarios[1, 1])
for i in 2:CANTIDAD_DE_AGENTES
  agentes[i] = Agente(NO_INFECTADO, horarios[1, i])
end

for i in 1:CANTIDAD_DE_ITERACIONES
  actualizar_posiciones_de_agentes(horarios, i, agentes)
  (indices_de_inicio_de_habitaciones, cantidad_de_agentes_por_habitacion, cantidad_de_agentes_infectados_por_habitacion) = asignar_inicios_de_habitaciones(agentes)

  agentes_ordenados = crear_lista_de_ubicaciones(indices_de_inicio_de_habitaciones, agentes)

  inicio = 1
  sum = 0
  for r in cantidad_de_agentes_por_habitacion
    if r != 0
      sum += r

      # iteramos sobre cada agente en cada habitacion
      for t in inicio:sum
        este_agente = agentes[agentes_ordenados[t]]
        if este_agente.estado == NO_INFECTADO && infectado(ubicaciones[este_agente.ubicacion], cantidad_de_agentes_infectados_por_habitacion[este_agente.ubicacion])
          print("agente #", agentes_ordenados[t] , " se infecto!\n")
          agentes[agentes_ordenados[t]] = Agente(INFECTADO, este_agente.ubicacion)
        end
      end

      inicio = r
    end
  end

  print("###############\n")

  mostrar_resultados(agentes, agentes_ordenados, i, cantidad_de_agentes_infectados_por_habitacion, ubicaciones)
end
