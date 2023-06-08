module infecciones_habitantes

using Plots: get_linewidth
using Plots, Colors
using Plots.PlotMeasures
using JLD2

include("./definiciones.jl")
include("./helper.jl")

function gaussian_dist(x)
  return (1 / sqrt(2 * pi)) * exp(-0.5 * (x - 1)^2)
end

function correr_simulacion()
  # las ubicaciones estan definidas solamente por sus tamanos
  # el primero es la casa del alumno, el segundo un corredor y el tercero es el comedor
  ubicaciones = Array{Float32, 1}(rand(Float32, CANTIDAD_DE_UBICACIONES) .* 10 .+ 10)
  ubicaciones[2] = 100 # corredor
  ubicaciones[3] = 80 # comedor

  horarios = ones(UInt32, (ITERACIONES_POR_DIA, CANTIDAD_DE_AGENTES))

  crear_horarios(horarios)
  agentes = Array{Agente, 2}(undef, (CANTIDAD_DE_ITERACIONES + 1, CANTIDAD_DE_AGENTES))

  agentes[1, 1] = Agente(INFECTADO, 0)
  for agente in 2:CANTIDAD_DE_AGENTES
    agentes[1, agente] = Agente(SUSCEPTIBLE, 0)
  end

  for dia in ProgressBar(0:(DIAS_DE_SIMULACION - 1))
    for iteracion in 1:ITERACIONES_POR_DIA
      iterador_global                          = (dia * ITERACIONES_POR_DIA) + iteracion
      agentes_en_ubicaciones                   = zeros(CANTIDAD_DE_UBICACIONES + 1)
      agentes_infectados_en_ubicacion_por_hora = zeros(CANTIDAD_DE_UBICACIONES + 1)

      # contamos los agentes
      for agenteID in 1:CANTIDAD_DE_AGENTES
        # Habra una probabilidad en forma de una poisson con un λ=1
        # para decidir cuando deberian llegar los alumnos

        if agentes[iterador_global, agenteID].estado != EN_CUARENTENA
          agentes_en_ubicaciones[horarios[iteracion, agenteID]] += 1
          if agentes[iterador_global, agenteID].estado == INFECTADO
            agentes_infectados_en_ubicacion_por_hora[horarios[iteracion, agenteID]] += 1
          end
        end
      end

       # infectar agentes
      for agente in 1:CANTIDAD_DE_AGENTES
        ubicacion = horarios[iteracion, agente]

        # por ahora solo hay dos estados en los que podria estar, por lo que parece
        # que esto se deberia poder simplificar pero voy a agregar mas en el futuro
        es_susceptible = agentes[iterador_global, agente].estado == SUSCEPTIBLE
        es_infectado = agentes[iterador_global, agente].estado == INFECTADO
        esta_en_cuarentena = agentes[iterador_global, agente].estado == EN_CUARENTENA
        es_recuperado = agentes[iterador_global, agente].estado == RECUPERADO

        no_esta_en_casa = ubicacion != 1
        se_deberia_infectar = rand(Float64) * 500 < agentes_infectados_en_ubicacion_por_hora[ubicacion] * (agentes_en_ubicaciones[ubicacion] / ubicaciones[ubicacion])

        agente_actual = agentes[iterador_global, agente]

        if es_susceptible && no_esta_en_casa && se_deberia_infectar
          agentes[iterador_global + 1, agente] = Agente(INFECTADO, 0)

        elseif es_infectado
          if agente_se_deberia_curar(agente_actual)
            agentes[iterador_global + 1, agente] = Agente(RECUPERADO, 0)
          elseif agente_se_deberia_cuarentenar(agente_actual)
            agentes[iterador_global + 1, agente] = Agente(EN_CUARENTENA, 0)
          end

        elseif esta_en_cuarentena && agente_se_deberia_curar(agente_actual)
          agentes[iterador_global + 1, agente] = Agente(RECUPERADO, 0)

        elseif es_recuperado
          if agente_actual.contador > TIEMPO_DE_RECUPERACION
            agentes[iterador_global + 1, agente] = Agente(SUSCEPTIBLE, 0)
          else
            agentes[iterador_global + 1, agente] = Agente(RECUPERADO, agentes[iterador_global + 1, agente] + 1)
          end

        else
          agentes[iterador_global + 1, agente] = agentes[iterador_global, agente]

        end
      end

    end

    es_fin_de_semana = dia % 5 == 0

    if es_fin_de_semana
      fin_de_semana.(agentes)
    else
      agente_en_casa.(agentes)
    end
  end

  save_object("agentes.jld2",
    Dict("agentes" => agentes, "horarios" => horarios)
  )
end

function graficar_simulacion()
  graficar_simulacion()
end

function graficar_horarios()
  horarios = load("agentes.jld2")["single_stored_object"]["horarios"]
  graficar_horarios(horarios)
end

end # module infecciones_habitantes
