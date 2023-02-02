using Base: iscontiguous, allocatedinline
using LinearAlgebra, OpenCL
using Plots
using Distributions
using BenchmarkTools

HACER_PLOT_DE_INFECTADOS = false
GRAFICAR_AGENTES = true

# defino un struct vector en vez de usar uno externo
# porque quiero tener control sobre la coherencia de los
# bits con el struct de la gpu
struct Vec2
  x::Float32
  y::Float32
end

struct Agent
  pos::Vec2
  vecl::Vec2
  state::UInt8
  iterations_spent_next_to_infected::UInt32
  iterations_spent_infected::UInt32
end

NO_INFECTADO = 0
INFECTADO = 1
RECUPERADO = 2

n = 10_000; # number of agents
borders = Array{Float32, 1}(undef, 2)
borders[1] = 500.
borders[2] = 500.
# sizeOfCuadrant = 10.

agents = Array{Agent, 1}(undef, n)

# Inicializar los valores de los agentes
for i in 1:n
  vels = rand(Uniform(-1, 1), 2)
  positions = rand(Float32, 2)
  positions[1] = positions[1] * borders[1]
  positions[2] = positions[2] * borders[2]

  if i < 10 
    agents[i] = Agent(Vec2(positions[1], positions[2]), Vec2(vels[1], vels[2]), INFECTADO, 0, 0)
  else
    agents[i] = Agent(Vec2(positions[1], positions[2]), Vec2(vels[1], vels[2]), NO_INFECTADO, 0, 0)
  end
end

kernel = read("./agents.cl", String)
device, ctx, queue = cl.create_compute_context()
p = cl.Program(ctx, source=kernel) |> cl.build!

agent_buff = cl.Buffer(Agent, ctx, (:r, :copy), hostbuf=agents)

update_agents_function =    cl.Kernel(p, "update_agent")
count_infected_agents_function = cl.Kernel(p, "count_infected_agents")

num_iterations = 500
infected_per_iteration = Array{UInt32, 1}(undef, num_iterations)
fill!(infected_per_iteration, 0)
infectedCount_buff = cl.Buffer(UInt32, ctx, (:r, :copy), hostbuf=infected_per_iteration)

if HACER_PLOT_DE_INFECTADOS
  print("Iniciando iteraciones\n")
  @time for i in 1:num_iterations
    queue(update_agents_function,    size(agents), nothing, agent_buff, convert(UInt32, n), convert(UInt32, i), borders[1], borders[2])
    queue(count_infected_agents_function, size(agents), nothing, agent_buff, infectedCount_buff, convert(UInt32, i))
  end
end

f = cl.read(queue, infectedCount_buff)
for i in 1:num_iterations
  infected_per_iteration[i] = f[i]
  print.(infected_per_iteration[i], " ")
end

if GRAFICAR_AGENTES
  # Cosas para graficar las posiciones de los agentes
  anim = @time @animate for i in 1:num_iterations
    queue(update_agents_function, size(agents), nothing, agent_buff, convert(UInt32, n), convert(UInt32, i))

    r = cl.read(queue, agent_buff)

    xs =     Array{Float32, 1}(undef, n)
    ys =     Array{Float32, 1}(undef, n)
    colors = Array{Symbol,  1}(undef, n)

    for i in eachindex(r)
      xs[i] = r[i].pos.x
      ys[i] = r[i].pos.y

      if r[i].state == 1
        colors[i] = :red
      elseif r[i].state == 0
        colors[i] = :blue
      elseif r[i].state == 2
        colors[i] = :green
      end

    end

    scatter(xs, ys, lab="", xlim=(-10, 110), ylim=(-10, 110), mc=colors)
  end

  gif(anim, "./gaming_time.gif", fps=60)
end

plot(0:num_iterations - 1, infected_per_iteration, label="infectados")
