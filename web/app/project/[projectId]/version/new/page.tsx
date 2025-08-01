'use client'

import { useState } from 'react'
import Sidebar from '@/components/Sidebar'
import StepWrapper from '@/components/StepWrapper'
import PreprocessingStepCard from '@/components/PreprocessingStepCard'
import { useVersionStore } from '@/store/useVersionStore'

export default function CreateVersionPage() {
  const { preprocessingSteps, addStep, removeStep } = useVersionStore()
  
  // 假设从数据源获取的图片数量
  const totalImages = 31
  
  // 初始化双滑块的值：slider1 控制训练集比例，slider2 控制训练集+测试集的比例
  // 初始值示例：训练集 60% ，测试集 30% ，验证集 10%
  const [slider1, setSlider1] = useState(60) // 训练集比例
  const [slider2, setSlider2] = useState(90) // 训练集+测试集比例，需确保 slider2 >= slider1

  // 计算比例
  const trainingSetCount = Math.round(totalImages * slider1 / 100)
  const testingSetCount = Math.round(totalImages * (slider2 - slider1) / 100)
  const validationSetCount = totalImages - trainingSetCount - testingSetCount

  // 处理滑块变化，确保 slider2 不低于 slider1
  const handleSlider1Change = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newVal = Number(e.target.value)
    setSlider1(newVal)
    if (newVal > slider2) {
      setSlider2(newVal)
    }
  }

  const handleSlider2Change = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newVal = Number(e.target.value)
    // 保证 slider2 不低于 slider1
    if (newVal < slider1) {
      setSlider1(newVal)
    }
    setSlider2(newVal)
  }

  return (
    <div className="flex min-h-screen bg-gray-50">
      <Sidebar />
      <main className="flex-1 p-8 space-y-8">
        <h1 className="text-3xl font-bold mb-6">Create New Version</h1>

        <StepWrapper step={1} title="Source Images">
          <p className="text-gray-600">
            Images: {totalImages} | Classes: 2 | Unannotated: 0
          </p>
        </StepWrapper>

        <StepWrapper step={2} title="Train/Test/Validation Split">
          <div className="space-y-4">
            <div>
              <label className="block text-gray-700 mb-2">
                Training Set: {slider1}%
              </label>
              <input
                type="range"
                min="0"
                max="100"
                value={slider1}
                onChange={handleSlider1Change}
                className="w-full"
              />
            </div>
            <div>
              <label className="block text-gray-700 mb-2">
                Training + Testing: {slider2}%
              </label>
              <input
                type="range"
                min="0"
                max="100"
                value={slider2}
                onChange={handleSlider2Change}
                className="w-full"
              />
            </div>
            <div className="text-gray-600">
              <p>Training Set: {trainingSetCount}</p>
              <p>Testing Set: {testingSetCount}</p>
              <p>Validation Set: {validationSetCount}</p>
            </div>
          </div>
        </StepWrapper>

        <StepWrapper step={3} title="Preprocessing">
          <div className="space-y-2">
            {preprocessingSteps.map((step, idx) => (
              <PreprocessingStepCard
                key={idx}
                step={step}
                onRemove={() => removeStep(idx)}
              />
            ))}
            <button
              className="px-4 py-2 bg-indigo-500 text-white rounded hover:bg-indigo-600"
              onClick={() =>
                addStep({ name: 'Resize', params: { width: 640, height: 640 } })
              }
            >
              + Add Preprocessing Step
            </button>
          </div>
        </StepWrapper>
      </main>
    </div>
  )
}