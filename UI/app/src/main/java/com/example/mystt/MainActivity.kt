package com.example.mystt

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.mystt.databinding.ActivityMainBinding

private lateinit var binding: ActivityMainBinding

class MainsActivity : AppCompatActivity() {
    private lateinit var speechRecognizer: SpeechRecognizer

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        val view = binding.root
        setContentView(view)

        // 권한 설정
        requestPermission()

        // RecognizerIntent 생성
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH)
        intent.putExtra(RecognizerIntent.EXTRA_CALLING_PACKAGE, packageName)    // 여분의 키
        intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE, "ko-KR")         // 언어 설정

        // <말하기> 버튼 눌러서 음성인식 시작
        binding.btnSpeech.setOnClickListener {
            // 새 SpeechRecognizer 를 만드는 팩토리 메서드
            speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this@MainActivity)
            speechRecognizer.setRecognitionListener(recognitionListener)    // 리스너 설정
            speechRecognizer.startListening(intent)                         // 듣기 시작
        }

    }

    // 권한 설정 메소드
    private fun requestPermission() {
        // 버전 체크, 권한 허용했는지 체크
        if (Build.VERSION.SDK_INT >= 23 &&
            ContextCompat.checkSelfPermission(this@MainActivity, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(this@MainActivity,
                arrayOf(Manifest.permission.RECORD_AUDIO), 0)
        }
    }

    // 리스너 설정
    private val recognitionListener: RecognitionListener = object : RecognitionListener {
        // 말하기 시작할 준비가되면 호출
        override fun onReadyForSpeech(params: Bundle) {
            Toast.makeText(applicationContext, "음성인식 시작", Toast.LENGTH_SHORT).show()
            binding.tvState.text = "이제 말씀하세요!"
        }
        // 말하기 시작했을 때 호출
        override fun onBeginningOfSpeech() {
            binding.tvState.text = "잘 듣고 있어요."
        }
        // 입력받는 소리의 크기를 알려줌
        override fun onRmsChanged(rmsdB: Float) {}
        // 말을 시작하고 인식이 된 단어를 buffer에 담음
        override fun onBufferReceived(buffer: ByteArray) {}
        // 말하기를 중지하면 호출
        override fun onEndOfSpeech() {
            binding.tvState.text = "끝!"
        }
        // 오류 발생했을 때 호출
        override fun onError(error: Int) {
            val message = when (error) {
                SpeechRecognizer.ERROR_AUDIO -> "오디오 에러"
                SpeechRecognizer.ERROR_CLIENT -> "클라이언트 에러"
                SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "퍼미션 없음"
                SpeechRecognizer.ERROR_NETWORK -> "네트워크 에러"
                SpeechRecognizer.ERROR_NETWORK_TIMEOUT -> "네트웍 타임아웃"
                SpeechRecognizer.ERROR_NO_MATCH -> "찾을 수 없음"
                SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> "RECOGNIZER 가 바쁨"
                SpeechRecognizer.ERROR_SERVER -> "서버가 이상함"
                SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> "말하는 시간초과"
                else -> "알 수 없는 오류임"
            }
            binding.tvState.text = "에러 발생: $message"
        }
        // 인식 결과가 준비되면 호출
        override fun onResults(results: Bundle) {
            // 말을 하면 ArrayList에 단어를 넣고 textView에 단어를 이어줌
            val matches = results.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
            for (i in matches!!.indices) binding.textView.text = matches[i]
        }
        // 부분 인식 결과를 사용할 수 있을 때 호출
        override fun onPartialResults(partialResults: Bundle) {}
        // 향후 이벤트를 추가하기 위해 예약
        override fun onEvent(eventType: Int, params: Bundle) {}
    }
}