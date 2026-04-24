---
theme: apple-basic
title: Estimadores de Contracción para Matrices de Precisión Cosmológicas
info: |
  Andrés F. Movilla Obregón
  Universidad del Norte, 24 de abril de 2026
layout: center
class: intro-slide text-center
highlighter: shiki
drawings:
  persist: false
mdc: true
transition: slide-left
fonts:
  sans: Inter
  serif: Source Serif Pro
  mono: Fira Code
---

# Implementación de Estimadores de Contracción para Matrices de Precisión Cosmológicas

<div class="pt-6 opacity-80">
Andrés F. Movilla Obregón<br/>
Tutor: Elias D. Niño-Ruiz, Ph.D.<br/>
Universidad del Norte · 24 de abril de 2026
</div>

---

# Objetivos

<div class="text-sm">

**General**

Evaluar y extender los estimadores de contracción directa de precisión en el régimen $n \sim N_e$, común al EnKF secuencial y al análisis de covarianza cosmológica.

**Específicos**

1. **Implementar** cuatro objetivos de precisión directa (identidad, valores propios, cosmológico e identidad escalada) dentro del marco TEDA.
2. **Comparar** contra Ledoit–Wolf, NERCOME y los baselines EnKF estándar y EnKF-Cholesky sobre Lorenz-96 y mocks cosmológicos de BOSS.
3. **Caracterizar** cómo la estructura del objetivo (diagonal vs. densa) afecta los coeficientes, el recorte y el compromiso sesgo–varianza.
4. **Proponer** un objetivo de identidad escalada agnóstico al dominio, como punto medio entre la identidad plana y el objetivo cosmológico específico.

</div>

---

# Cómo Funciona el EnKF

En cada ciclo $t$:

1. **Simular** $N_e$ pronósticos paralelos (el ensamble) propagando el modelo hacia adelante
2. **Estimar** una matriz de covarianza ($\mathbf{B}_t$) a partir de esos pronósticos
3. **Invertirla** para obtener una matriz de precisión ($\mathbf{B}_t^{-1}$)
4. **Usar** la matriz de precisión para combinar el pronóstico con las observaciones reales

La matriz de precisión se **reestima y reutiliza en cada ciclo**.

---

# El Paralelo Cosmológico

Para un análisis:

1. **Simular** $N_{\text{mocks}}$ catálogos mock (realizaciones sintéticas del universo)
2. **Estimar** una matriz de covarianza ($\boldsymbol{\Sigma}$) a partir de esos mocks
3. **Invertirla** para obtener una matriz de precisión ($\boldsymbol{\Sigma}^{-1}$)
4. **Usar** la matriz de precisión en una verosimilitud gaussiana para ajustar parámetros contra la medición real

La matriz de precisión se **estima una vez y se reutiliza como fija**.

---

# Misma Matriz, Nombres Distintos

El paso 2 en ambos flujos calcula el mismo tipo de matriz.

<div class="text-sm pt-2">

|                                | Asimilación de Datos             | Cosmología                         |
| ------------------------------ | -------------------------------- | ---------------------------------- |
| Muestras                       | $N_e$ pronósticos del ensamble   | $N_{\text{mocks}}$ catálogos mock  |
| Matriz de covarianza de …      | error de pronóstico              | medición del espectro de potencia  |
| Nombre de la matriz            | $\mathbf{B}_t$                   | $\boldsymbol{\Sigma}$              |

</div>

**Mismo objeto estadístico:** una matriz de covarianza muestral estimada a partir de $N$ muestras de un vector $n$-dimensional.

De aquí en adelante, la llamaremos $\mathbf{S}$: la _matriz de covarianza muestral genérica_.

---

# La Matriz de Covarianza Muestral Genérica

Para $N$ muestras $\mathbf{x}_1, \dots, \mathbf{x}_N$:

$$\mathbf{S} = \frac{1}{N - 1}\sum_{i=1}^{N}(\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})^T$$

**Correcta en promedio**, pero para cualquier estimación individual:

- **Muy ruidosa cuando $N \sim n$**: el ruido escala como $n^2/N$
- **No invertible cuando $N \leq n$**: la matriz es singular
- **La inversión amplifica el ruido**: valores propios pequeños en $\mathbf{S}$ se vuelven enormes en $\mathbf{S}^{-1}$

Así que en el régimen de $N$ pequeño y $n$ grande, $\mathbf{S}$ es inutilizable. Necesitamos un mejor estimador.

---

# Sesgo vs. Varianza

El error total de cualquier estimador se descompone en dos partes:

$$\text{Error total} \ = \ \text{Sesgo}^2 \ + \ \text{Varianza}$$

- **Sesgo**: desviación sistemática respecto a la verdad
- **Varianza**: fluctuación aleatoria entre distintos conjuntos de muestras

$\mathbf{S}$ tiene **sesgo cero** pero **varianza enorme**.

**La idea:** aceptar un pequeño sesgo a cambio de mucha menos varianza.

---

# Contracción (Shrinkage)

Combinar $\mathbf{S}$ con una matriz objetivo estructurada $\mathbf{T}$:

$$\hat{\mathbf{S}} \ = \ (1 - \lambda)\,\mathbf{S} \ + \ \lambda\,\mathbf{T}$$

- $\mathbf{T}$: algo en lo que confiamos a priori (p. ej. la identidad, o una matriz informada por el dominio)
- $\lambda \in [0, 1]$: cuánto confiar en $\mathbf{T}$ frente a $\mathbf{S}$
  - $\lambda = 0$: $\mathbf{S}$ pura
  - $\lambda = 1$: $\mathbf{T}$ pura

Los "mejores estimadores" del área se diferencian en **cómo eligen $\mathbf{T}$ y $\lambda$**.

---

# Estado del Arte

<div class="text-sm">

| Familia                     | Representante                         | Qué hace                                             |
| --------------------------- | ------------------------------------- | ---------------------------------------------------- |
| Covarianza lineal           | Ledoit–Wolf (2004)                    | Combina $\mathbf{S}$ con una identidad escalada      |
| Sin objetivo                | NERCOME (2016)                        | Promedia muchas estimaciones de valores propios por bootstrap |
| No lineal                   | Ledoit–Wolf (2012, 2020)              | Usa un $\lambda$ distinto para cada valor propio     |
| Precisión esparcida         | GLasso, CLIME                         | Fuerza a cero la mayoría de entradas de $\mathbf{S}^{-1}$ |
| **Precisión directa**       | **Bodnar et al. (2016)**              | Combina $\mathbf{S}^{-1}$ directamente (nuestra base) |
| Objetivo cosmológico        | Pope–Szapudi (2008), Looijmans (2024) | Usa una matriz objetivo específica del dominio       |
| Localización                | Gaspari–Cohn; Cholesky modificado     | Anula correlaciones de largo alcance en $\mathbf{S}$ |

</div>

### Pero cada uno tiene una debilidad dominante

---

# Comparación Crítica

Cuatro debilidades transversales a las siete familias:

1. **Objeto incorrecto.** La mayoría contrae $\mathbf{S}$, pero necesitamos $\mathbf{S}^{-1}$. Invertir una $\mathbf{S}$ contraída amplifica errores en valores propios pequeños.

2. **Demasiado costosas.** NERCOME hace ~200 bootstraps por llamada. Aceptable una vez, prohibitivo dentro de un bucle EnKF de 100 ciclos.

3. **Forma incorrecta.** Los objetivos diagonales fallan cuando la precisión verdadera es densa (p. ej. Lorenz-96); los coeficientes explotan y se recortan cada ciclo.

4. **Métrica incorrecta.** Los métodos minimizan el error de una sola estimación. Pero el EnKF reutiliza el estimador en cada ciclo, así que los errores se acumulan.

---

# La Brecha de Investigación

Ningún método existente cumple **los cuatro** a la vez:

1. Contrae la matriz de **precisión** (no solo la de covarianza)
2. **Suficientemente barato** para correr en cada ciclo dentro de un EnKF
3. **Estable** a lo largo de ciclos secuenciales (baja varianza)
4. **Objetivo flexible**: funciona sin conocimiento del dominio


---

# Método Propuesto

Partimos de Bodnar et al. (2016). En lugar de contraer $\mathbf{S}$ y luego invertir, **contraemos $\mathbf{S}^{-1}$ directamente** contra una matriz objetivo de precisión $\boldsymbol{\Pi}_0$:

$$\boldsymbol{\Pi}_{\text{LS}} \ = \ \alpha^*\,\mathbf{S}^{-1} \ + \ \beta^*\,\boldsymbol{\Pi}_0$$

- $\alpha^*$, $\beta^*$: pesos analíticos, derivados para minimizar el error esperado de $\boldsymbol{\Pi}_{\text{LS}}$ (forma cerrada, recalculados a partir de los datos cada vez)
- $\boldsymbol{\Pi}_0$: una matriz objetivo de precisión; elección nuestra

Esto **corrige la debilidad #1** de la diapositiva anterior: ahora contraemos el objeto que realmente necesitamos.

---

# Contribución 1: Acoplamiento Secuencial

**Incrustamos la contracción directa de precisión dentro del bucle EnKF.** En cada ciclo $t$:

1. Tomamos el ensamble actual
2. Calculamos $\mathbf{S}$ y recalculamos $\alpha^*, \beta^*$ **desde cero**
3. Formamos $\boldsymbol{\Pi}_{\text{LS}}$ y lo alimentamos a la ganancia de Kalman

**Ningún artículo existente hace esto para contracción directa de precisión.** \
Bodnar et al. (2016) y Looijmans et al. (2024) la aplican solo a estimación cosmológica de una sola estimación.

---

# Cuatro Opciones de Objetivo Evaluadas

| Objetivo                  | $\boldsymbol{\Pi}_0$                                  | Notas                                  |
| ------------------------- | ----------------------------------------------------- | -------------------------------------- |
| Identidad                 | $\mathbf{I}$                                          | valor por defecto genérico             |
| Valores propios           | usa los valores propios de $\mathbf{S}^{-1}$          | se adapta al espectro de los datos     |
| Cosmológico               | $\text{diag}(N_\ell / 2C_\ell^2)$                     | incorpora conocimiento del dominio     |
| **Identidad escalada**    | $\text{diag}(2 / \mathbf{S}_{ii})$                    | **nuevo en esta tesis**                |

A continuación, una diapositiva corta por cada objetivo.

---

# Objetivo: Identidad

$$\boldsymbol{\Pi}_0 = \mathbf{I}$$

- El **valor por defecto genérico**. Sin suposiciones sobre los datos
- Trata a todas las variables como si estuvieran en la misma escala
- **Falla** cuando las variables tienen magnitudes muy distintas (p. ej. el espectro de potencia a $\ell$ bajo vs. alto)

<div class="text-xs opacity-60 pt-3">Objetivo base en Bodnar et al. (2016); análogo a la identidad escalada de Ledoit–Wolf (2004).</div>

---

# Objetivo: Valores Propios

$$\boldsymbol{\Pi}_0 = \mathbf{U}\,\text{diag}(\lambda_i)\,\mathbf{U}^T$$

donde $\mathbf{U}$, $\lambda_i$ son los vectores propios y valores propios de $\mathbf{S}^{-1}$.

- **Se adapta al espectro de los propios datos**. Sin parámetros externos
- Conserva la parte diagonal de $\mathbf{S}^{-1}$ en su propia base propia
- Autorreferencial: usa $\mathbf{S}^{-1}$ para construir su propio objetivo

<div class="text-xs opacity-60 pt-3">Introducido por Bodnar et al. (2016) como objetivo alternativo para la contracción directa de precisión.</div>

---

# Objetivo: Cosmológico

$$\boldsymbol{\Pi}_0 = \text{diag}\!\left(\frac{N_\ell}{2\,C_\ell^2}\right)$$

- $C_\ell$: espectro de potencia teórico en el multipolo $\ell$, obtenido de teoría de perturbaciones cosmológicas y parámetros de referencia
- $N_\ell \approx (2\ell+1)\,f_{\text{sky}}$: conteo de modos a partir de la geometría del relevamiento
- Asume que el campo de densidad subyacente es **gaussiano**

**Requiere una cosmología fija, un relevamiento específico y una aproximación gaussiana.** \
No se transfiere a otros dominios.

<div class="text-xs opacity-60 pt-3">Pope & Szapudi (2008) lo usaron primero para contracción de covarianza en cosmología; Looijmans et al. (2024) lo aplicó en el marco de precisión directa de Bodnar.</div>

---

# Un Hueco en el Menú de Objetivos

Cada objetivo existente tiene una debilidad:

- **Identidad**: ignora la escala de cada variable
- **Valores propios**: **autorreferencial** y **operacionalmente costoso** (usa $\mathbf{S}^{-1}$ para construir su propio objetivo; duplica el costo por ciclo)
- **Cosmológico**: **requiere conocimiento del dominio** (una cosmología y relevamiento específicos)

Falta: un objetivo que sea **consciente de la escala**, **simple** y **agnóstico al dominio**.\
Construido a partir de un resumen barato de los datos, no de la inversa ruidosa completa.

---

# Contribución 2: Objetivo de Identidad Escalada

$$(\boldsymbol{\Pi}_0)_{ii} \ = \ \frac{2}{\mathbf{S}_{ii}}$$

- **Consciente de la escala**: varianzas por variable tomadas de la diagonal de $\mathbf{S}$ misma
- **Agnóstico al dominio**: sin parámetros externos, sin cosmología, sin relevamiento
- **Simple**: solo lee la diagonal de $\mathbf{S}$; $\mathcal{O}(n)$ por ciclo
- El factor 2 refleja la escala cosmológica $2/N_\ell$

**Llena el hueco:** consciente de la escala + agnóstico al dominio + simple.

---

# Por Qué Funciona en la Práctica

Consecuencias del diseño consciente de la escala + agnóstico + simple:

- **Objetivo con magnitud pareada**: la diagonal $2/\mathbf{S}_{ii}$ ya está en el orden de magnitud correcto respecto a $\mathbf{S}^{-1}$, así que $\alpha^*$ y $\beta^*$ permanecen en $[0, 1]$ con mucha más frecuencia que con la identidad plana → **menor tasa de recorte**.
- **Basado en datos pero barato**: a diferencia del objetivo de valores propios (duplica el costo por ciclo) o del cosmológico (requiere configuración previa), es una lectura diagonal de una sola línea.

Resultado: **mejora del 10–30%** sobre la identidad plana en ambos bancos de prueba (RMSE en DA, pérdida de Frobenius en cosmología).

---

# Banco de Prueba 1: Lorenz-96

Un modelo de juguete caótico de dinámica atmosférica. Dimensión del estado $n = 40$.

- Forzamiento $F = 8$ (fuertemente caótico), integración Runge-Kutta de 4º orden
- Observaciones: 32 de 40 variables observadas, ruido gaussiano $\sigma_o = 0.01$
- Factor de inflación $\rho = 1.05$
- Ventana de asimilación: 100 ciclos
- Repeticiones: **30 corridas** por configuración

Las variables vecinas están acopladas por la dinámica → **la matriz de precisión verdadera es densa fuera de la diagonal**.\
El caso difícil para objetivos diagonales.

---

# Banco de Prueba 2: Modelo AR(1) Cosmológico

Un modelo dinámico simplificado cuyas estadísticas estacionarias coinciden con los mocks cosmológicos BOSS DR12. Dimensión del estado $d = 18$ (9 números de onda × 2 multipolos).

- Persistencia $\phi = 0.95$ (evolución temporal suave), calibrada a 2048 mocks Patchy
- La covarianza estacionaria iguala por construcción la covarianza de los mocks BOSS
- Observaciones: 14 de 18 variables observadas, ruido gaussiano $\sigma_o = 100$
- Factor de inflación $\rho = 1.02$
- Repeticiones: **30 corridas** por configuración

Los modos cosmológicos son casi independientes → **la matriz de precisión verdadera es casi diagonal**.\
El caso fácil para objetivos diagonales.

---

# ¿Por Qué Ambos Bancos de Prueba?

Los dos sistemas tienen **estructura de precisión opuesta** por diseño:

- **Lorenz-96**: precisión densa fuera de la diagonal (caso difícil)
- **AR(1) cosmológico**: precisión casi diagonal (caso fácil)

Correr los **mismos métodos** en ambos permite separar dos efectos:

- ¿Funciona el **método** en sí?
- ¿El **objetivo coincide con la verdad**?

Un solo banco confundiría ambos efectos.

---

# Ocho Métodos Comparados

Enfrentamiento directo en ambos bancos:

- **Referencias**: EnKF, EnKF-Cholesky
- **Contracción de covarianza**: Ledoit–Wolf
- **Sin objetivo**: NERCOME
- **Contracción directa de precisión** (nuestro marco), con cuatro objetivos:
  - Identidad
  - Valores propios
  - Cosmológico *
  - **Identidad escalada** (nuevo en esta tesis)

<div class="text-xs opacity-60 pt-4">
* El objetivo cosmológico no puede construirse en Lorenz-96 porque depende de cantidades específicas del dominio cosmológico.<br/>
Evaluado solo en el banco cosmológico.
</div>

---

# Tamaño del Ensamble: Cota Inferior

**¿Qué tan pequeño puede ser $N_e$?** Dos restricciones:

- **$N_e > n + 2$**: necesaria para que $\mathbf{S}$ sea invertible
- **$N_e > n + 4$**: necesaria para que los coeficientes de Bodnar ($\alpha^*$, $\beta^*$) estén bien definidos

Para Lorenz-96 ($n = 40$), eso significa $N_e \geq 44$. Nuestro valor más pequeño probado es $N_e = 50$, seis miembros por encima de la cota.

Cualquier cosa menor cae en el **régimen submuestreado** que esta tesis no cubre.

---

# Tamaño del Ensamble: Cota Superior

**¿Qué tan grande puede ser $N_e$ antes de que la contracción pierda sentido?**

A medida que $N_e \gg n$:

- El ruido en $\mathbf{S}$ se reduce (escala como $n^2 / N_e$)
- El peso de contracción $\lambda$ cae hacia $0$
- $\mathbf{S}$ ya es buena, el objetivo deja de importar

Más allá de $N_e \approx 3n$, la contracción no ofrece ganancia significativa sobre $\mathbf{S}$ plana.

Para Lorenz-96 paramos en $N_e = 100$ ($n / N_e = 0.4$), bien dentro del régimen interesante.

---

# Tamaños de Ensamble Probados: Lorenz-96

Dimensión del estado $n = 40$.

| $N_e$ | $n/N_e$ | Régimen y rol                                                    |
| ----- | ------- | ---------------------------------------------------------------- |
| 50    | 0.80    | Mal condicionado. Exigencia en el límite de invertibilidad       |
| 60    | 0.67    | Moderadamente mal condicionado                                   |
| 80    | 0.50    | Moderadamente mal condicionado                                   |
| 100   | 0.40    | Moderado → frontera de buen muestreo                             |

Cuatro valores elegidos para abarcar el régimen entre la cota inferior y la superior.

---

# Tamaños de Ensamble Probados: Cosmológico

Dimensión del estado $d = 18$.

| $N_e$ | $n/N_e$ | Régimen y rol                                                    |
| ----- | ------- | ---------------------------------------------------------------- |
| 24    | 0.75    | Mal condicionado. Coincide con presupuestos operacionales de mocks |
| 30    | 0.60    | Moderado (punto de trabajo de Looijmans et al.)                  |
| 40    | 0.45    | Moderado                                                         |
| 50    | 0.36    | Moderado → frontera de buen muestreo                             |

Validado empíricamente con un barrido más fino de 12 valores de $N_e$ (en el apéndice). \
Sin transiciones abruptas entre estos puntos.

---
layout: image-right
image: ./figures/rmse_vs_ne.png
---

# RMSE vs $N_e$ en Lorenz-96

<div class="text-sm">

- Métodos de precisión directa: **erráticos**, RMSE 0.5–1.8 × 10⁻²
- Ledoit–Wolf: **menor** RMSE, estable en todo $N_e$
- EnKF-Cholesky: competitivo gracias a la estructura bandeada
- EnKF estándar: diverge con $N_e$ pequeño

**Causa:** precisión densa + objetivos diagonales → miscalibración severa de los coeficientes.

</div>

---

# ¿Por Qué Fallan los Métodos de Precisión Directa en Lorenz-96?

Miremos los coeficientes de contracción:

- $\alpha^*$ se mantiene dentro de su rango válido, cerca de la cota teórica $1 - n/N_e$.
- $\beta^*$ **explota a $10^8$–$10^9$**, muy fuera del rango válido $[0, 1]$, así que se recorta de vuelta a $1$.
- **El recorte ocurre en > 80% de los ciclos** en los tres objetivos diagonales.

Resultado: la matriz de precisión $\boldsymbol{\Pi}_{\text{LS}}$ queda muy mal condicionada (número de condición $10^7$–$10^{11}$, vs. $10^2$–$10^3$ para Ledoit–Wolf).

**Causa raíz: precisión densa + objetivo diagonal = desajuste forzado.** La fórmula de Bodnar intenta corregir inflando $\beta^*$; el recorte impide la corrección.

---
layout: image-right
image: ./figures/beta_vs_ne.png
---

# $\beta^*$ sin Recortar Explota

<div class="text-sm">

$|\beta^*|$ sin recortar vs $N_e$ (escala log). Línea punteada: umbral de recorte en $\beta^* = 1$ (rango válido $[0, 1]$).

- **Objetivo identidad:** $10^7$–$10^9$ veces demasiado grande
- **Identidad escalada:** mejor, pero aún requiere recorte
- **Objetivo valores propios:** el único que se comporta normalmente aquí

</div>

---
layout: image-right
image: ./figures/cosmo_enkf_rmse.png
---

# Dominio Cosmológico

<div class="text-sm">

- Todos los métodos son estables, **reducción monótona del RMSE** con $N_e$
- Precisión directa con objetivo cosmológico → menor pérdida de Frobenius
- Identidad escalada: segundo mejor método agnóstico al dominio
- **Ledoit–Wolf gana en RMSE dentro del EnKF**

</div>

---

# El Hallazgo Central

<div class="text-center text-3xl pt-8 font-bold">
Mejor estimador aislado ≠ mejor filtro EnKF.
</div>

<div class="text-center text-lg pt-6 opacity-80">
Una desconexión no documentada previamente en la literatura de contracción.
</div>

---

# Aislado vs Secuencial (Dominio Cosmológico)

**Estimación de precisión aislada** (pérdida de Frobenius de una sola estimación):

- Precisión directa con **objetivo cosmológico** gana, pérdida $\approx 0.47$
- Otros objetivos de precisión directa: valores propios > identidad escalada > identidad (todos $\approx 0.54$)
- Ledoit–Wolf peor, pérdida $\approx 0.93$–$0.96$

**Filtrado EnKF** (RMSE sobre 100 ciclos):

- Ledoit–Wolf **gana**
- Precisión directa con objetivo cosmológico: segundo
- Otros objetivos de precisión directa: identidad escalada > valores propios > identidad \
(el nuestro es el mejor entre los agnósticos al dominio)


---

# Por Qué se Invierte el Ranking

El EnKF aplica el estimador en cada ciclo, así que la **varianza se acumula**.

- Desv. estándar de Ledoit–Wolf entre sorteos: **$\pm 0.01$**
- Desv. estándar de precisión directa: **$\pm 0.13$–$0.14$** (un orden de magnitud mayor)

En 100 ciclos, una brecha de varianza de 100× supera la ventaja en sesgo.

**Principio:** baja varianza le gana a bajo sesgo cuando los errores se acumulan secuencialmente.

---

# Costo Asintótico

<div class="text-xs">

| Método                                                         | Costo por ciclo                  |
| -------------------------------------------------------------- | -------------------------------- |
| EnKF                                                           | $\mathcal{O}(n^2 N_e + m^2 N_e)$ |
| EnKF-Cholesky                                                  | $\mathcal{O}(n \, r \, N_e)$     |
| Precisión directa (identidad / identidad escalada / cosmológico) | $\mathcal{O}(n^2 N_e + n^3)$     |
| Precisión directa (objetivo valores propios)                   | $\mathcal{O}(n^2 N_e + 2n^3)$    |
| NERCOME                                                        | $\mathcal{O}(B(n^2N_e + n^3))$   |
| Ledoit–Wolf                                                    | $\mathcal{O}(n^2 N_e)$           |

</div>

<div class="text-xs opacity-75">

$n$: dimensión del estado · $N_e$: tamaño del ensamble · $m$: observaciones · $r$: ancho de banda de Cholesky · $B$: repeticiones bootstrap (~200)
</div>

---

# Conclusiones sobre el Costo

- **Precisión directa**: añade una inversión $\mathcal{O}(n^3)$ sobre lo que el EnKF ya hace
- **NERCOME**: multiplica el costo de precisión directa por $B$ (~200 repeticiones bootstrap)
- **EnKF-Cholesky**: el único método **lineal en $n$**; el único que escala a dimensiones operacionales

---

# Tiempo de Reloj Medido

**Lorenz-96** ($n = 40$, $N_e = 80$, por 100 ciclos):

- Todos los métodos de contracción: 26–28 s
- NERCOME: **215 s**, 8× más lento

**Cosmológico** ($d = 18$, $N_e = 30$, por 100 ciclos):

- Métodos de contracción: 0.20–0.28 s
- NERCOME: **26.7 s**, 190× más lento
- Referencia EnKF: 0.14 s

La propagación del ensamble domina en Lorenz-96; la estimación de precisión domina en cosmología.

---
layout: image-right
image: ./figures/timing_scaling.png
---

# Escalado a Dimensiones Operacionales

<div class="text-sm">

Extrapolación desde los tiempos medidos usando fórmulas asintóticas:

- **$n \lesssim 100$:** el costo no diferencia
- **$n \sim 10^2$–$10^3$:** la inversión $n^3$ domina; NERCOME queda descartado
- **$n \gtrsim 10^4$:** solo EnKF-Cholesky (lineal en $n$) sobrevive

**Implicación práctica:** la contracción directa de precisión es más adecuada para bancos de prueba de escala media o combinada con localización a escala operacional.

</div>

---

# Recomendación Práctica

<div class="text-sm pt-2">

| Régimen ($n/N_e$)       | Estructura de precisión | ¿Objetivo conocido? | Recomendación                                        |
| ----------------------- | ----------------------- | ------------------- | ---------------------------------------------------- |
| $>1$ (submuestreado)    | cualquiera              | n/a                 | Localización + EnKF-Cholesky                         |
| $0.5$–$1$               | casi diagonal           | **sí**              | Precisión directa + objetivo del dominio             |
| $0.5$–$1$               | casi diagonal           | **no**              | **Identidad escalada** (este trabajo) o Ledoit–Wolf  |
| $0.5$–$1$               | densa                   | n/a                 | Ledoit–Wolf; EnKF-Cholesky si $n \gtrsim 10^3$       |
| $<0.3$ (bien muestreado)| cualquiera              | n/a                 | EnKF plano (el peso de contracción $\lambda \to 0$)  |

</div>

<br/>

La precisión guía la elección dentro de la banda $\mathcal{O}(n^3)$; el costo la guía en los extremos.

---

# Contribuciones

1. **Primer acoplamiento secuencial** de la contracción directa de precisión de Bodnar, con coeficientes recalculados en cada ciclo dentro del EnKF.

2. **Nuevo objetivo de identidad escalada**, un punto medio basado en datos entre la identidad plana y el objetivo cosmológico específico del dominio.

3. **Banco de prueba dual**, sobre sistemas con estructura de precisión opuesta, separando el diseño del método de la alineación objetivo-verdad.

4. **Aislado ≠ secuencial**, el hallazgo clave: la varianza del estimador importa tanto como el sesgo cuando los errores se acumulan.

5. **Estudio sistemático del tamaño del ensamble**, documentando cuándo y cómo fallan los coeficientes de contracción.

6. **Implementación de código abierto** en el marco TEDA para los seis estimadores.

---

# Limitaciones

- **Régimen submuestreado no cubierto**: cuando $N_e < n$, $\mathbf{S}$ es singular y todos los métodos de precisión directa fallan. Requeriría localización.
- **Bancos de prueba de baja dimensión**: $n = 40$ y $d = 18$ son minúsculos comparados con DA operacional ($10^6$–$10^8$).
- **Modelo cosmológico simplificado**: AR(1) es lineal; los relevamientos reales involucran crecimiento no lineal de estructura.
- **Solo operadores de observación lineales**: sin pruebas con modelos de medición no lineales.
- **Factor heurístico de 2** en el objetivo escalado: escogido por analogía con cosmología; podría validarse de forma cruzada.

---

# Trabajo Futuro: Metodología

- **Contracción × localización**: combinar ambas estrategias de regularización
- **Sistemas de escala intermedia**: probar el escalado en modelos cuasi-geostróficos ($n \sim 10^4$)
- **Selección adaptativa del objetivo**: cambiar de objetivo según la estructura fuera de la diagonal de los datos
- **Análisis teórico de la acumulación secuencial**: derivar rigurosamente el compromiso sesgo/varianza

---

# Trabajo Futuro: Extensiones del Estimador

- **Factor de escala con fundamento**: reemplazar el 2 heurístico por un valor validado de forma cruzada
- **Muestras no gaussianas**: extender la derivación de Bodnar más allá de ensambles gaussianos
- **Cota sobre el número de condición**: truncar valores propios diminutos para evitar el estallido de la inversión
- **Objetivo adaptativo en el tiempo**: actualizar el objetivo a partir de una media móvil de estimaciones previas

---
layout: center
class: intro-slide text-center
---

# Gracias


<div class="pt-6 opacity-70 text-sm">
Andrés F. Movilla Obregón · Universidad del Norte<br/>
Tutor: Elias D. Niño-Ruiz, Ph.D.
</div>
