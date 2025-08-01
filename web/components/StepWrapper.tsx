type Props = {
  step: number
  title: string
  children: React.ReactNode
}

export default function StepWrapper({ step, title, children }: Props) {
  return (
    <div className="mb-6">
      <div className="flex items-center mb-2">
        <div className="w-6 h-6 rounded-full bg-indigo-500 text-white text-sm flex items-center justify-center mr-2">
          {step}
        </div>
        <h2 className="text-lg font-semibold">{title}</h2>
      </div>
      <div className="ml-8">{children}</div>
    </div>
  )
}
