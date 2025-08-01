type Props = {
  step: { name: string; params?: Record<string, any> }
  onRemove: () => void
}

export default function PreprocessingStepCard({ step, onRemove }: Props) {
  return (
    <div className="border px-4 py-2 rounded flex justify-between items-center bg-white shadow-sm">
      <div>
        <strong>{step.name}</strong>
        {step.params && (
          <span className="text-gray-600 ml-2 text-sm">
            ({Object.keys(step.params).map(k => `${k}: ${step.params![k]}`).join(', ')})
          </span>
        )}
      </div>
      <button className="text-red-500 hover:underline" onClick={onRemove}>
        Remove
      </button>
    </div>
  )
}
