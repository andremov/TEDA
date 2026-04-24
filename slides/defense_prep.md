# Preparación para la defensa

MSc · Universidad del Norte · viernes 24 de abril de 2026, 8:00–10:00 am

---

## Qué pasa el día de la defensa

**Formato típico (2 h):**

1. **Presentación del estudiante** — 20–30 min. Exposición del trabajo sin interrupciones.
2. **Preguntas y comentarios del jurado** — 45–60 min. El tutor modera. Cada jurado pregunta en turnos; puede haber back-and-forth.
3. **Deliberación privada** — el jurado pide que salgas (o te desconectes si es virtual), 5–15 min.
4. **Resultado** — te llaman y comunican aprobación/no aprobación y la nota. A veces se firma el acta en ese momento.

**Vestimenta:** formal (saco, camisa; corbata no obligatoria). Criterio de entrevista formal.

**Llegada:** 20 min antes. Verifica proyector, HDMI, audio, laptop en modo presentación, PDF exportado en USB como respaldo.

---

## Preguntas probables

Basadas en el tema y en las observaciones del jurado ya emitidas.

### Sobre método

- **¿Por qué shrinkage de precisión directa y no Ledoit-Wolf + inversión?** — respondido en §5.7.1 (covarianza ≠ precisión).
- **¿Qué tan robusta es la asunción Wishart bajo no-linealidad del modelo?** — Lorenz-96 es caótico, los ensambles no son exactamente gaussianos; reconocido en limitaciones.
- **¿Por qué clampean tanto los coeficientes?** — target-truth mismatch en Lorenz. Punto central del discussion.

### Sobre experimentos

- **¿Por qué $N_e \in \{50,60,80,100\}$?** — §1.3 nueva lista la justificación.
- **¿Por qué solo 30 corridas?** — intervalos de confianza razonables con menos costo. Verificado con IC.
- **¿Por qué no se probó el régimen $N_e < n$?** — Bodnar requiere invertibilidad; limitación explícita.

### Sobre alcance y validez

- **¿Se generaliza a $n$ grande?** — §7.2 de extrapolación; cruce con Cholesky en $n \sim 10^3$–$10^4$.
- **¿Por qué no se sometió a revista?** — paper completo, no sometido por tiempo; trabajo futuro.
- **Defender el hallazgo "varianza > sesgo en filtrado secuencial".** — empírico; análisis vía Lyapunov propuesto como trabajo futuro.

### Sobre contribución

- **¿Cuál es la contribución más fuerte?** — el hallazgo standalone ≠ secuencial, no el target escalado (incremental).
- **¿Por qué el factor 2 en el target escalado?** — analogía cosmológica, reconocido como heurístico.

---

## Plan de preparación

### Hoy (miércoles, previo al viernes)

1. **Ensayo cronometrado** de la presentación 2 veces completas. Objetivo: 22–25 min. Si pasa de 28, recortar. Grabar en video una corrida — ayuda a detectar muletillas.

2. **Memorizar 3 números clave** para tenerlos en la punta de la lengua:
   - Clamping > 80% en Lorenz-96.
   - $|\beta^*|$ alcanza $10^8$–$10^9$ sin clamping.
   - NERCOME 190× más lento en cosmo, 8× en Lorenz.

3. **Lista escrita de 10 preguntas duras** con respuesta de 2–3 frases cada una. Impresa, no mental — bajo estrés el papel funciona.

4. **Limitaciones fluidas.** Cuando el jurado señale una, reconócela rápido y conecta con trabajo futuro. No defender lo indefendible. *"Sí, es una limitación — lo abordaríamos con X."*

5. **PDF de respaldo** de los slides en USB y en correo personal. Tesis impresa o en tablet por si piden consultar una ecuación específica.

### Mañana antes de la defensa

- Desayuno liviano. Café moderado (no triples).
- Llegar a las 7:30. Probar el proyector antes del inicio.
- Agua en la mesa.
- Si preguntan algo que no sabes: *"Esa parte no la exploré en profundidad, pero mi intuición es X — valdría la pena verificarlo."* Mejor que inventar.

---

## Tácticas durante el Q&A

- **Pausa antes de responder.** 2 segundos de silencio se sienten eternos para ti pero al jurado le parecen reflexivos.
- **Repite la pregunta** con tus palabras antes de responder. Te da tiempo y asegura que la entendiste: *"Si entiendo bien, me preguntas sobre..."*
- **No interrumpas** al jurado. Nunca.
- *"Esa es una buena observación"* es aceptable una vez, no tres veces.
- Si te corrigen algo factual: *"Tienes razón, lo anoto para la versión final."*
- Al cierre: agradecer al tutor y a los jurados por su tiempo y retroalimentación.

---

## Específico a este caso

- Las 3 observaciones del jurado ya están incorporadas — mencionarlo al inicio del Q&A si es relevante: *"Antes de empezar las preguntas, las observaciones del comité previo fueron integradas en secciones §1.3, §5.7.1 y §7."*
- El target escalado es la contribución más original pero más débil. **Defenderlo por lo que es:** middle-ground domain-agnostic, no un salto metodológico.
- El hallazgo central (standalone ≠ sequential) es más fuerte conceptualmente — ahí está el mayor valor.
- Es una defensa de MSc, no PhD. No hay que probar novedad radical; hay que demostrar dominio técnico y pensamiento crítico.

---

## Checklist final

### Materiales
- [ ] Slides en laptop (ambos idiomas si aplica)
- [ ] Slides en PDF en USB
- [ ] Slides en PDF en correo personal
- [ ] Tesis en PDF accesible (laptop + tablet/impresa)
- [ ] Cable HDMI + adaptador
- [ ] Cargador laptop
- [ ] Agua

### Administrativos
- [ ] Invitación con salón + foto
- [ ] Ficha de cumplimiento (esperar formato de coordinación)
- [x] Turnitin enviado
- [x] Resumen para invitación

### Mentales
- [ ] 10 preguntas duras con respuestas escritas
- [ ] 2 ensayos completos cronometrados
- [ ] Revisar §1.3, §5.7.1, §7.1–§7.4 (secciones nuevas)
- [ ] Dormir 7+ horas jueves en la noche
