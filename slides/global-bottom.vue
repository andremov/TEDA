<script setup>
import { useNav } from '@slidev/client'
const { currentPage, total } = useNav()
</script>

<template>
  <div class="page-counter">{{ currentPage }} / {{ total }}</div>
</template>

<style scoped>
.page-counter {
  position: absolute;
  bottom: 0.75rem;
  right: 1.25rem;
  font-size: 0.75rem;
  opacity: 0.5;
  font-variant-numeric: tabular-nums;
  letter-spacing: 0.02em;
  pointer-events: none;
  z-index: 10;
}
</style>
