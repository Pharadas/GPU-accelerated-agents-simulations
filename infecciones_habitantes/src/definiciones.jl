using Distributions
using StatsBase
using ProgressBars

const HORAS_EN_EL_DIA = 24
const DIAS_DE_SIMULACION = 90
const HORAS_DE_SIMULACION = 15
const TIEMPO_POR_ITERACION = 1 / 4
const ITERACIONES_POR_DIA = convert(Int, HORAS_DE_SIMULACION / TIEMPO_POR_ITERACION)
const CANTIDAD_DE_ITERACIONES = convert(Int, ITERACIONES_POR_DIA * DIAS_DE_SIMULACION)
const CANTIDAD_DE_AGENTES = 100
const CANTIDAD_DE_UBICACIONES = 50
const HORAS_DE_CLASE = 2
const ITERACIONES_POR_CLASE = HORAS_DE_CLASE / TIEMPO_POR_ITERACION

const TIEMPO_DE_RECUPERACION = 120
const TIEMPO_POR_FIN_DE_SEMANA = HORAS_EN_EL_DIA * 2
const TIEMPO_ENTRE_DIAS = HORAS_EN_EL_DIA - HORAS_DE_SIMULACION

const ALUMNOS_SON_LENTOS = true;
const GRAFICAR = false
const CORRER_SIMULACION = false
const GRAFICAR_HORARIOS = true

const SUSCEPTIBLE::UInt8 = 0;
const INFECTADO::UInt8 = 1;
const EN_CUARENTENA::UInt8 = 2;
const RECUPERADO::UInt8 = 3;

mutable struct Agente
  estado::UInt8
  contador::UInt32
end