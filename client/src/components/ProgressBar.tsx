interface ProgressBarProps {
  progress: number;
}

const ProgressBar: React.FC<ProgressBarProps> = ({ progress }) => {
  return (
    <div className="w-full bg-gray-200 h-4 rounded mt-2">
      <div
        className="bg-green-500 h-4 rounded transition-all duration-500 ease-out"
        style={{ width: `${progress}%` }}
      />
      <p className="text-sm text-gray-700 mt-1">{progress}%</p>
    </div>
  );
};

export default ProgressBar;
