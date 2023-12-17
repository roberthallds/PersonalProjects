//
//  ContentView.swift
//  First App
//
//  Created by Robert Hall on 11/9/23.
//

import SwiftUI

struct ContentView: View {
    
    @State private var creditScore = 1000;
    @State private var slotOne = "apple"
    @State private var slotTwo = "cherry"
    @State private var slotThree = "star"
    
    let slotHolder = ["apple", "cherry", "star"]
    
    var body: some View {
        VStack {
            //Image(systemName: "globe")
               // .imageScale(.large)
               // .foregroundColor(.accentColor)
            Text("SwiftUI Slots!")
                .font(.title)
                .padding()
            Spacer()
            Group{Text("Credits: ") + Text(String(creditScore))}
                .font(.title2)
            Spacer()
            HStack {
                Image("apple").resizable()
                Image("cherry").resizable()
                Image("star").resizable()
            }.aspectRatio(contentMode: .fit)
            Spacer()
            Button(action: {
                slotOne = slotHolder.randomElement()!
                slotTwo = slotHolder.randomElement()!
                slotThree = slotHolder.randomElement()!
                if (slotOne == slotTwo && slotTwo == slotThree) {
                    creditScore += 35
                } else {
                    creditScore -= 5
                }
            }, label: {
                Text("Spin").padding([.leading, .trailing], 40)
                    .foregroundColor(.white)
                    .background(Color(.systemPink))
                    .cornerRadius(25)
                    .font(.system(size: 25, weight: .bold, design: .default))
            })
            Spacer()
        }
        .padding()
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
