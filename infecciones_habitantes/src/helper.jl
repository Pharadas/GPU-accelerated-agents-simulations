include("./definiciones.jl")

function fin_de_semana(agente::Agente)
  # TODO: probabilidad de infectarse / desinfectarse a lo largo del fin de semana

  return Agente(agente.estado, agente.contador + TIEMPO_POR_FIN_DE_SEMANA)
end

function agente_en_casa(agente::Agente)
  # TODO: probabilidad de infectarse / curarse estando en casa

  return Agente(agente.estado, agente.contador + TIEMPO_ENTRE_DIAS)
end

function crear_horarios(horarios)
  td = truncated(Poisson(1.5), 0.0, Inf)
  for agente in ProgressBar(1:CANTIDAD_DE_AGENTES)
    # vamos a decir que cada alumno tiene 3 clases de dos horas
    # las cuales pueden empezar en las siguientes horas:
    # 0, 2, 4, 6, 10, 12
    posibles_horarios = [0, 2, 4, 8, 10, 12]
    clases_por_dia = 5

    horarios_de_clases = sample(1:length(posibles_horarios), Weights(pdf.(td, 1:length(posibles_horarios))), clases_por_dia; replace=false)
    horarios_de_clases = sort(horarios_de_clases)
    salones_de_clase = rand(4:CANTIDAD_DE_UBICACIONES, clases_por_dia)

    tiempo_real = 0
    curr_class = 0
    comiendo = false
    ya_comio = false
    tiempo_que_lleva_comiendo = 0

    for iteracion in 1:ITERACIONES_POR_DIA
      if curr_class == 0 # antes de que llegue a clase
        if tiempo_real >= horarios_de_clases[1]
          curr_class = 1
        end

      else # una vez que esta en el campus
        if tiempo_real > horarios_de_clases[clases_por_dia] + HORAS_DE_CLASE # si ya termino todas sus clases deberia regresar a su casa
          break

        elseif tiempo_real > horarios_de_clases[curr_class] + HORAS_DE_CLASE # si ya termino esta clase deberia ir a la siguiente
            curr_class += 1
        end

        # si esta dentro del horario de una clase, deberia estar en esa clase
        if tiempo_real >= horarios_de_clases[curr_class] && tiempo_real <= horarios_de_clases[curr_class] + HORAS_DE_CLASE
          horarios[iteracion, agente] = salones_de_clase[curr_class]

        # si no ha comido tiene una probabilidad de comer,
        # tambien si ya esta comiendo deberia continuar
        elseif (!ya_comio && rand() > 0.3) || comiendo
          comiendo = true
          horarios[iteracion, agente] = 3
          tiempo_que_lleva_comiendo += TIEMPO_POR_ITERACION
        end

        # si lleva mucho tiempo comiendo deberia dejar de hacerlo
        if tiempo_que_lleva_comiendo > 1.5
          comiendo = false
          ya_comio = true

        else
          # si no esta comiendo deberia estar en el pasillo
          # TODO: hacer que haya una lista de pasillos no solo uno
          horarios[iteracion, agente] = 2
        end
      end

      tiempo_real += TIEMPO_POR_ITERACION
    end
  end
end

function agente_se_deberia_curar(agente::Agente)
  return rand() / (agente.contador * 0.5) < 0.0001
end

function agente_se_deberia_cuarentenar(agente::Agente)
  return rand() < 0.5
end

function graficar_simulacion()
  t = -2.5mm
  size_t = ceil(Int, sqrt(CANTIDAD_DE_UBICACIONES))

  agentes = load("agentes.jld2")["single_stored_object"]["agentes"]
  horarios = load("agentes.jld2")["single_stored_object"]["horarios"]
  # graficacion de la informacion
  # infectados = zeros(CANTIDAD_DE_ITERACIONES)

  color_coding = Array{Symbol, 1}(undef, 3)
  color_coding[NO_INFECTADO + 1] = :blue
  color_coding[INFECTADO + 1] = :red

  infectados = []
  @gif for i in ProgressBar(1:CANTIDAD_DE_ITERACIONES)
    colors = []
    agentsPosX = []
    agentsPosY = [] 
    infectados_iteracion = 0

    for agente in 1:CANTIDAD_DE_AGENTES
      ubic = horarios[i % ITERACIONES_POR_DIA + 1, agente] - 1
      initial_pos = [(floor(Int, ubic % size_t) + rand() % 0.8) + 0.1, (floor(Int, ubic / size_t) + rand() % 0.8) + 0.1]

      push!(agentsPosX, initial_pos[1])
      push!(agentsPosY, initial_pos[2])

      push!(colors, color_coding[agentes[i, agente].estado + 1])

      infectados_iteracion += agentes[i, agente].estado == INFECTADO
    end

    push!(infectados, infectados_iteracion)
    # infectados[i] = infectados_iteracion
    # display(infectados)

    scatter(agentsPosX, agentsPosY, markercolor=colors, markersize = 3)
    first_plot = plot!(positionsX, positionsY, legends=false)
    second_plot = scatter(1:i, infectados, markersize = 3)
    plot(first_plot, second_plot, layout=2) 
  end

  # plot(1:CANTIDAD_DE_ITERACIONES, infectados)
  # savefig("./gaming_time.png")
end

function graficar_horarios(horarios)
  agentes_en_escuela_por_dia = zeros(ITERACIONES_POR_DIA)

  for iteracion in 1:ITERACIONES_POR_DIA
    agentes_en_escuela = 0

    for agenteID in 1:CANTIDAD_DE_AGENTES
      if horarios[iteracion, agenteID] != 1
        agentes_en_escuela += 1
      end
    end

    agentes_en_escuela_por_dia[iteracion] = agentes_en_escuela
  end

  plot(agentes_en_escuela_por_dia)
end
