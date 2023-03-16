using GLMakie: ColorTypes
using Base: iscontiguous, allocatedinline
using LinearAlgebra, OpenCL
using GLMakie
using Distributions
using BenchmarkTools
using JLD2
using YAML
using ProgressBars

# defino un struct Vec2 en vez de usar uno externo
# porque quiero tener control sobre la coherencia de los
# bits con el struct de la gpu
struct Vec2
  x::Float32
  y::Float32
end

# Esto se debe poder reducir mas, peor por ahora deberia
# ser suficiente
struct Agent
  pos::Vec2
  vel::Vec2
  state::UInt8
  counter::UInt8
end

struct SimulationInfo
  amountOfAgents::UInt32
  iteration::UInt32
  sizeOfSector::UInt32
  borders::Vec2
end

data = YAML.load_file("simulation_data.yml")

for i in data
  println(i.first, " => ", i.second)
end

# Tres estados en los que se puede encontrar un agente,
# convertido a variable por claridad
SUSCEPTIBLE = 0
INFECTADO_ASINTOMATICO = 1
INFECTADO_SINTOMATICO = 2
RECUPERADO = 3
EN_CUARENTENA = 4

n = data["numero_de_agentes"] # numero de agentes en la iteracion
num_iterations = data["numero_de_iteraciones"]
borders = Array{Float32, 1}(data["bordes"]) # bordes de la simulacion
size_of_cuadrant::UInt32 = data["tamano_de_division_de_seccion"]
initial_infected = data["infectados_iniciales"]

kernel = read("./SEIIQR.cl", String)
device, ctx, queue = cl.create_compute_context()
p = cl.Program(ctx, source=kernel) |> cl.build!

# Hacer link con las funciones del codigo de OpenCL
update_agents_function                           = cl.Kernel(p, "update_agent")
clear_buffer                                     = cl.Kernel(p, "clear_buffer")
update_filled_cuadrants                          = cl.Kernel(p, "update_filled_cuadrants")
assign_agents_to_cuadrants                       = cl.Kernel(p, "assign_agents_to_cuadrants")
calcular_posiciones_de_cuadriculas               = cl.Kernel(p, "calcular_posiciones_de_cuadriculas")
two_pass_consecutive_vector_addition_first_pass  = cl.Kernel(p, "two_pass_consecutive_vector_addition_first_pass")
two_pass_consecutive_vector_addition_second_pass = cl.Kernel(p, "two_pass_consecutive_vector_addition_second_pass")

function main()
  GLMakie.set_window_config!(framerate = Inf, vsync = false)
  simInfo = SimulationInfo(n, 0, size_of_cuadrant, Vec2(borders[1], borders[2]))

  amount_of_cuadrants::UInt32 = trunc(UInt32, (((borders[1]) / size_of_cuadrant) + 2) * (((borders[2]) / size_of_cuadrant) + 2))

  # Nuestra lista de agentes es para cada punto de la iteracion
  agents               = Array{Agent , 1}(undef, n * num_iterations)
  cuadrants            = Array{UInt32, 1}(undef, n)
  cuadrants_filled     = Array{UInt32, 1}(undef, amount_of_cuadrants)
  cuadrants_counter    = Array{UInt32, 1}(undef, amount_of_cuadrants)
  cuadrants_semaphores = Array{UInt32, 1}(undef, amount_of_cuadrants)

  fill!(cuadrants, 0)
  fill!(cuadrants_filled, 0)
  fill!(cuadrants_counter, 0)
  fill!(cuadrants_semaphores, 0)
  fill!(agents, Agent(Vec2(0., 0.), Vec2(0., 0.), SUSCEPTIBLE, 0))

  # print("\ntotal bytes sent to gpu: ", (sizeof(agents) + sizeof(cuadrants) + sizeof(cuadrants_filled) + sizeof(cuadrants_counter) + sizeof(cuadrants_semaphores)), "\n")
  print("Creating data for ", n, " agents during ", num_iterations, " iterations\n")

  # Inicializar los valores de los agentes
  for i in 1:n
    vels = rand(Uniform(-1, 1), 2)
    positions = rand(Float32, 2)
    positions[1] = positions[1] * borders[1]
    positions[2] = positions[2] * borders[2]

    # decidir cuantos agentes empiezan infectados
    if i < initial_infected
      # agents[i] = Agent(Vec2(positions[1], positions[2]), Vec2(vels[1], vels[2]), INFECTADO, 0, 0)
      agents[i] = Agent(Vec2(positions[1], positions[2]), Vec2(vels[1], vels[2]), INFECTADO_ASINTOMATICO, 0)
      # agents[i] = Agent(Vec2(positions[1], positions[2]), Vec2(0., 0.), INFECTADO, 0)
    else
      # agents[i] = Agent(Vec2(positions[1], positions[2]), Vec2(vels[1], vels[2]), NO_INFECTADO, 0, 0)
      agents[i] = Agent(Vec2(positions[1], positions[2]), Vec2(vels[1], vels[2]), SUSCEPTIBLE, 0)
    end
  end

  # Crear buffers en la memoria de la gpu y copiar los contenidos
  # de nuestros arrays
  agent_buff                = cl.Buffer(Agent,  ctx, (:r, :copy), hostbuf=agents)
  cuadrants_buff            = cl.Buffer(UInt32, ctx, (:r, :copy), hostbuf=cuadrants)
  cuadrants_filled_buff     = cl.Buffer(UInt32, ctx, (:r, :copy), hostbuf=cuadrants_filled)
  cuadrants_counter_buff    = cl.Buffer(UInt32, ctx, (:r, :copy), hostbuf=cuadrants_counter)
  cuadrants_semaphores_buff = cl.Buffer(UInt32, ctx, (:r, :copy), hostbuf=cuadrants_semaphores)

  print("Running simulation\n")
  @time for i in ProgressBar(1:(num_iterations - 1))
    simInfo = SimulationInfo(n, i, size_of_cuadrant, Vec2(borders[1], borders[2]))
    # Clean up all the necessary buffers on the GPU
    queue(clear_buffer, n,                   nothing, cuadrants_buff)
    queue(clear_buffer, amount_of_cuadrants, nothing, cuadrants_filled_buff)
    queue(clear_buffer, amount_of_cuadrants, nothing, cuadrants_counter_buff)
    queue(clear_buffer, amount_of_cuadrants, nothing, cuadrants_semaphores_buff)

    queue(update_filled_cuadrants,            n, nothing, agent_buff, cuadrants_buff, cuadrants_filled_buff, simInfo)
    queue(calcular_posiciones_de_cuadriculas, 1, nothing, cuadrants_filled_buff, amount_of_cuadrants)
    queue(assign_agents_to_cuadrants,         n, nothing, agent_buff, cuadrants_buff, cuadrants_filled_buff, cuadrants_counter_buff, cuadrants_semaphores_buff, simInfo)

    queue(update_agents_function,
          n,
          nothing,
          agent_buff,
          cuadrants_buff,
          cuadrants_filled_buff,
          simInfo,
          convert(UInt32, amount_of_cuadrants),
      )

    # print(i, "\r")
  end

  # Leer la lista de agentes una vez que ya calculamos
  # todas las iteraciones
  agentes_procesado = cl.read(queue, agent_buff)

  print("Saving data to 'agentes.jld2'\n")
  save_object("agentes.jld2", agentes_procesado)

  cl.release!(agent_buff)
  cl.release!(cuadrants_buff)
  cl.release!(cuadrants_filled_buff)
  cl.release!(cuadrants_counter_buff)
  cl.release!(cuadrants_semaphores_buff)

  return nothing
end

function clear()
  cl.release!(queue)
  cl.release!(device)
  cl.release!(ctx)

  cl.release!(kernel)
end

function load_and_render()
  agentes_procesado = load("agentes.jld2")["single_stored_object"]

  # Una vez que tenemos estos agentes podemos hacer lo que queramos con ellos
  # lo siguiente es solo una manera de graficarlos y guardar esa grafica
  print("Starting rendering\n")
  points        = Observable(Point2[(0.0, 0.0)])
  c             = Observable([:black])

  fig = Figure()

  ax1 = Axis(fig[1, 1],
      title = "Simulation",
      xlabel = "The x label",
      ylabel = "The y label"
  )

  ax2 = Axis(fig[1, 2],
      title = "Infected",
      xlabel = "The x label",
      ylabel = "The y label"
  )

  limits!(ax1, 0, borders[1], 0, borders[2])
  limits!(ax2, 0, num_iterations, 0, n)

  scatter!(ax1, points, color=c, markersize = 2)
  # lines!(  ax2, infected; color=:red, linewidth=4)

  print("No cerrar la ventana hasta que se terminen de procesar los datos\n")
  record(fig, "Infected_Agents.gif", ProgressBar(1:num_iterations); framerate =  30) do i
    # como estamos leyendo una lista muy larga con muchas iteraciones
    # necesitamos un indice inicial por iteracion
    b = (i - 1) * n

    new_points = Array{Point2, 1}(undef, n)
    new_colors = Array{Symbol, 1}(undef, n)
    susceptible = 0
    infectado_sintomatico = 0
    infectado_asintomatico = 0
    recuperado = 0
    cuarentena = 0

    for agent in 1:n
      # guardar las posiciones de los agentes
      new_points[agent] = Point2(agentes_procesado[b + agent].pos.x, agentes_procesado[b + agent].pos.y)

      # decidir el color que vamos a dibujar en base a sus estados
      if agentes_procesado[b + agent].state == SUSCEPTIBLE
        new_colors[agent] = :green
        susceptible += 1

      elseif agentes_procesado[b + agent].state == INFECTADO_SINTOMATICO
        new_colors[agent] = :purple
        infectado_sintomatico += 1

      elseif agentes_procesado[b + agent].state == INFECTADO_ASINTOMATICO
        new_colors[agent] = :red
        infectado_asintomatico += 1

      elseif agentes_procesado[b + agent].state == RECUPERADO
        new_colors[agent] = :blue
        recuperado += 1

      elseif agentes_procesado[b + agent].state == EN_CUARENTENA
        new_colors[agent] = :yellow
        cuarentena += 1
      end
    end

    points[]        = new_points
    c[]             = new_colors

    scatter!(ax2, [i], [susceptible]; color=:green, markersize=2)
    scatter!(ax2, [i], [infectado_sintomatico]; color=:purple, markersize=2)
    scatter!(ax2, [i], [infectado_asintomatico]; color=:red, markersize=2)
    scatter!(ax2, [i], [recuperado]; color=:blue, markersize=2)
    scatter!(ax2, [i], [cuarentena]; color=:yellow, markersize=2)

    # lines!(ax2, iter_num, infected_count; color=:red, linewidth=4)

    # push!(infected[], Point2(convert(Float64, i), convert(Float64, infected_count)))

    # append!(infected[], infected_count)
    # append!(iteration_num[], i)
  end

  print("Puede cerrar la ventana")
end

t::UInt32 = trunc(UInt32, (((borders[1]) / size_of_cuadrant) + 2) * (((borders[2]) / size_of_cuadrant) + 2))

print("Estos son los parametros de la simulacion, se enviaran ", ((sizeof(Agent) * n * (num_iterations + 1)) + (t * sizeof(UInt32) * 3)) / 1_000_000, " megabytes a la gpu, continuar? y/n\n> ")
if (readline() == "y")
  println()
  main()
  print("\nQuiere graficar los resultados? y/n\n> ")
  if (readline() == "y")
    println()
    load_and_render()
  end

else
  print("Quiere graficar los resultados guardados? y/n\n> ")
  if (readline() == "y")
    load_and_render()
  end
end
