import { create } from 'zustand'

type PreprocessingStep = {
  name: string
  params?: Record<string, any>
}

interface VersionState {
  preprocessingSteps: PreprocessingStep[]
  addStep: (step: PreprocessingStep) => void
  removeStep: (index: number) => void
}

export const useVersionStore = create<VersionState>((set) => ({
  preprocessingSteps: [{ name: 'Auto-Orient' }],
  addStep: (step) => set((state) => ({
    preprocessingSteps: [...state.preprocessingSteps, step]
  })),
  removeStep: (index) => set((state) => ({
    preprocessingSteps: state.preprocessingSteps.filter((_, i) => i !== index)
  })),
}))
