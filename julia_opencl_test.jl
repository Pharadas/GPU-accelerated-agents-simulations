using GLMakie: ColorTypes
using Base: iscontiguous, allocatedinline
using LinearAlgebra, OpenCL
using GLMakie
using Distributions
using BenchmarkTools

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
  vecl::Vec2
  state::UInt8
  iterations_spent_next_to_infected::UInt32
  iterations_spent_infected::UInt32
end

# Tres estados en los que se puede encontrar un agente,
# convertido a variable por claridad
NO_INFECTADO = 0
INFECTADO = 1
RECUPERADO = 2

n = 10_000; # numero de agentes en la iteracion
num_iterations = 200
borders = Array{Float32, 1}(undef, 2) # bordes de la simulacion
borders[1] = 100.
borders[2] = 100.

# Nuestra lista de agentes es para cada punto de la iteracion
agents = Array{Agent, 1}(undef, n * num_iterations)

# Inicializar los valores de los agentes
for i in 1:n
  vels = rand(Uniform(-1, 1), 2)
  positions = rand(Float32, 2)
  positions[1] = positions[1] * borders[1]
  positions[2] = positions[2] * borders[2]

  # decidir cuantos agentes empiezan infectados
  if i < 10
    agents[i] = Agent(Vec2(positions[1], positions[2]), Vec2(vels[1], vels[2]), INFECTADO, 0, 0)
  else
    agents[i] = Agent(Vec2(positions[1], positions[2]), Vec2(vels[1], vels[2]), NO_INFECTADO, 0, 0)
  end
end

# Cargar el codigo y crear el contexto de OpenCL
kernel = read("./agents.cl", String)
device, ctx, queue = cl.create_compute_context()
p = cl.Program(ctx, source=kernel) |> cl.build!

# Hacer link con las funciones del codigo de OpenCL
update_agents_function = cl.Kernel(p, "update_agent")
count_infected_agents_function = cl.Kernel(p, "count_infected_agents")

# Crear buffers en la memoria de la gpu y copiar los contenidos
# de nuestros arrays
agent_buff = cl.Buffer(Agent, ctx, (:r, :copy), hostbuf=agents)

infected_per_iteration = Array{UInt32, 1}(undef, num_iterations)
fill!(infected_per_iteration, 0)
infectedCount_buff = cl.Buffer(UInt32, ctx, (:r, :copy), hostbuf=infected_per_iteration)

print("Iniciando iteraciones\n")
@time for i in 1:num_iterations
  queue(update_agents_function, n, nothing, agent_buff, convert(UInt32, n), convert(UInt32, i), borders[1], borders[2])
end

# Leer la lista de agentes una vez que ya calculamos
# todas las iteraciones
agentes_procesado = cl.read(queue, agent_buff)

# Una vez que tenemos estos agentes podemos hacer lo que queramos con ellos
# lo siguiente es solo una manera de graficarlos y guardar esa grafica

# creo los observables que cambiaran
# en cada frame
points = Observable(Point2[(0.0, 0.0)])
c =      Observable([RGBf0(0, 0, 0)])

fig, ax = scatter(points, color=c, markersize = 5)
limits!(ax, 0, borders[1], 0, borders[2])

print("No cerrar la ventana hasta que se terminen de proceas los datos\n")
record(fig, "append_animation.mp4", 1:num_iterations; framerate =  30) do i
  # como estamos leyendo una lista muy larga con muchas iteraciones
  # necesitamos un indice inicial por iteracion
  b = (i - 1) * n

  new_points = Array{Point2,         1}(undef, n)
  new_colors = Array{ColorTypes.RGB, 1}(undef, n)

  for agent in 1:n
    # guardar las posiciones de los agentes
    new_points[agent] = Point2(agentes_procesado[b + agent].pos.x, agentes_procesado[b + agent].pos.y)

    # decidir el color que vamos a dibujar en base a sus estados
    if agentes_procesado[b + agent].state == INFECTADO
      new_colors[agent] = RGBf0(1, 0, 0)
    elseif agentes_procesado[b + agent].state == NO_INFECTADO
      new_colors[agent] = RGBf0(0, 1, 0)
    elseif agentes_procesado[b + agent].state == RECUPERADO
      new_colors[agent] = RGBf0(0, 0, 1)
    end
  end

  points[] = new_points
  c[]      = new_colors
end

print("Puede cerrar la ventana")
